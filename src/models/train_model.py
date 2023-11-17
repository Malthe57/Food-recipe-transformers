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
from dataset_class import FoodRecipeDataset
from image_encoder import ImageEncoder, ResNetBackbone
from text_encoder import TextEncoder
from joint_encoder import JointEmbedding
import argparse

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataloaders(batch_size, pretrained=False, image_size=(224,224)):
    
    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer("basic_english")
    transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])

    if pretrained:
        VocabImage = FoodRecipeDataset(text_path, images_path, transform=transform)
    else:
        VocabImage = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=transform)
    
    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImage))
    test_size = len(VocabImage) - training_size
    generator = torch.Generator().manual_seed(42)
    training_data, temp = random_split(VocabImage, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)

    if pretrained:
        trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
    else:
        trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return trainloader, valloader, testloader, training_data, val_data, test_data


def main(image_size=(64,64), patch_size=(8,8), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4,pos_enc='learnable',
         pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=32, lr=3e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1, model_name = "../../models/best_model_ever.pt", only_title=True, pretrained=False
         
    ):

    writer = SummaryWriter()

    loss_function = TripletLoss(margin=0.1)

    trainloader, valloader, _, _, _, _ = prepare_dataloaders(batch_size=batch_size, pretrained=pretrained, image_size=image_size)

    if pretrained:
        image_encoder = ResNetBackbone(embed_dim=embed_dim)
        print("Using pretrained ResNet18 backbone")
    else:
        image_encoder = ImageEncoder(image_size=image_size, patch_size=patch_size, channels=channels, 
                    embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                    pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim)
        
    title_encoder = TextEncoder(embed_dim=embed_dim, num_heads=num_heads, num_layers = num_layers, max_seq_len = 512,
                dropout = dropout, fc_dim = fc_dim, num_tokens = 50000, pool = "mean", pos_enc = pos_enc, pretrained=pretrained)
    ingredients_encoder = TextEncoder(embed_dim=embed_dim, num_heads=num_heads, num_layers = num_layers, max_seq_len = 512,
                dropout = dropout, fc_dim = fc_dim, num_tokens = 50000, pool = "mean", pos_enc = pos_enc, pretrained=pretrained)
    instructions_encoder = TextEncoder(embed_dim=embed_dim, num_heads=num_heads, num_layers = num_layers, max_seq_len = 512,
                dropout = dropout, fc_dim = fc_dim, num_tokens = 50000, pool = "mean", pos_enc = pos_enc, pretrained=pretrained)
    
    model = JointEmbedding(image_encoder=image_encoder, title_encoder=title_encoder, ingredients_encoder=ingredients_encoder, 
                           instructions_encoder=instructions_encoder, embed_dim = embed_dim, only_title=only_title, pretrained=pretrained)

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {model_params}")

    image_params = sum(p.numel() for p in image_encoder.parameters())
    print(f"Total number of parameters in the image encoder: {image_params}")

    text_params = sum(p.numel() for p in title_encoder.parameters())
    print(f"Total number of parameters in the title encoder: {text_params}")

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(k / warmup_steps, 1.0))

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
                if pretrained:
                    image = image.to('cuda')
                else:
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
                    if pretrained:
                        image = image.to('cuda')
                    else:
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
    set_seed(seed=42)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--patch_size', type=int, default=8, help='patch size')
    parser.add_argument('--channels', type=int, default=3, help='number of channels')
    parser.add_argument('--embed_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--pos_enc', type=str, default='learnable', help='position encoding')
    parser.add_argument('--pool', type=str, default='cls', help='pooling')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--model_name', type=str, default='models/best_model_ever.pt', help='model name')
    parser.add_argument('--only_title', default=True, action='store_true') 
    parser.add_argument('--pretrained', default=False, action='store_true')   


    args = parser.parse_args()

    model = main(image_size=(args.image_size, args.image_size), patch_size=(args.patch_size, args.patch_size), 
                 model_name=args.model_name, lr=args.lr, num_epochs=args.num_epochs, only_title=args.only_title, 
                 pretrained=args.pretrained)


