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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from einops import rearrange
from torchtext.data.utils import get_tokenizer
import sys
sys.path.append("../Food-recipe-transformers/src/")
from utils.loss import TripletLoss

from VocabImagedataset_class import VocabImageDataset, pad_input, collate_batch
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from joint_encoder import JointEmbedding

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataloaders(batch_size):
    
    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer("basic_english")
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])

    VocabImageData = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=transform)


    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImageData))
    test_size = len(VocabImageData) - training_size
    generator = torch.Generator().manual_seed(42)
    training_data, temp = random_split(VocabImageData, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)

    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


    return trainloader, valloader, testloader, training_data, val_data, test_data


def main(image_size=(64,64), patch_size=(8,8), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1, model_name = "../../models/best_model_ever.pt"
         
    ):

    writer = SummaryWriter()

    loss_function = TripletLoss()

    trainloader, valloader, _, _, _, _ = prepare_dataloaders(batch_size=batch_size)


    image_encoder = ImageEncoder(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes)
    
    text_encoder = TextEncoder(embed_dim=embed_dim, num_heads=num_heads, num_layers = num_layers, max_seq_len = 512,
                   dropout = dropout, fc_dim = fc_dim, num_tokens = 50000, pool = "mean", pos_enc = pos_enc)
    model = JointEmbedding(image_encoder=image_encoder, text_encoder=text_encoder, embed_dim = embed_dim, only_title=True)

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {model_params}")

    image_params = sum(p.numel() for p in image_encoder.parameters())
    print(f"Total number of parameters in the image encoder: {image_params}")

    text_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"Total number of parameters in the text encoder: {text_params}")



    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    i = 0
    j = 0
    best_val_loss = np.inf
    for e in range(num_epochs):
        print(f'\n epoch {e}')

        train_losses = []

        model.train()
        for image, title, ingredients, instructions, cleaned_ingredients, _ in tqdm.tqdm(trainloader):
            if torch.cuda.is_available():
                image, title, ingredients, instructions, cleaned_ingredients = image.to('cuda'), title.to('cuda'), ingredients.to('cuda'), instructions.to('cuda'), cleaned_ingredients.to('cuda')
            opt.zero_grad()
            image_features, text_features = model(image, title, ingredients, instructions)
            loss = loss_function(image_features, text_features)

            train_losses.append(loss.item())

            writer.add_scalar("Train/Loss", loss, i)

            loss.backward()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

            i += 1

        print(f'-- {"train"} loss {np.mean(train_losses):.3}')

        with torch.no_grad():

            val_losses = []

            model.eval()

            for image, title, ingredients, instructions, cleaned_ingredients, _ in valloader:
                if torch.cuda.is_available():
                    image, title, ingredients, instructions, cleaned_ingredients = image.to('cuda'), title.to('cuda'), ingredients.to('cuda'), instructions.to('cuda'), cleaned_ingredients.to('cuda')
                image_features, text_features = model(image, title, ingredients, instructions)
                val_loss = loss_function(image_features, text_features)
                writer.add_scalar("Val/Loss", val_loss, j)

                val_losses.append(val_loss.item())

                j += 1

            if np.mean(val_losses) < best_val_loss:   
                torch.save(model, model_name)
                best_val_loss = np.mean(val_losses)
                print(f"New best model with val loss: {best_val_loss:.3}")
        

    return model

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)

    model_path = "models/best_model_ever.pt"
    model = main(image_size=(64, 64), patch_size=(8,8), model_name=model_path, lr=3e-4, num_epochs=200)
