import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from utils.loss import TripletLoss
import matplotlib.pyplot as plt
from einops import rearrange
from dataset_class import FoodRecipeDataset
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataloaders(batch_size, classes=[3, 7]):
    
    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "dataset/Food Images")
    text_path = os.path.join(current_working_directory, "dataset/food.csv")

    FoodRecipeData = FoodRecipeDataset(text_path, images_path, transforms.ToTensor())


    training_share = 0.6 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(FoodRecipeData))
    test_size = len(FoodRecipeData) - training_size
    generator = torch.Generator().manual_seed(42)
    training_data, test_data = random_split(FoodRecipeData, [training_size, test_size], generator)

    trainloader = DataLoader(training_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32, shuffle=True)

    return trainloader, testloader, training_data, test_data


def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1
         
    ):

    loss_function = TripletLoss()

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ImageEncoder(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )   


    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        for image, title, ingredients, instructions, cleaned_ingredients in tqdm.tqdm(train_iter):
            if torch.cuda.is_available():
                image, title, ingredients, instructions, cleaned_ingredients = image.to('cuda'), title.to('cuda'), ingredients.to('cuda'), instructions.to('cuda'), cleaned_ingredients.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, title)
            loss.backward()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for image, title, instructions, cleaned_ingredients in test_iter:
                if torch.cuda.is_available():
                    image, title, ingredients, instructions, cleaned_ingredients = image.to('cuda'), title.to('cuda'), ingredients.to('cuda'), instructions.to('cuda'), cleaned_ingredients.to('cuda')
                out = model(image).argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            print(f'-- {"validation"} accuracy {acc:.3}')
    return model

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    model = main()
    torch.save(model, 'ViT_model.pt')
