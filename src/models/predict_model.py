import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import pickle
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
from dataset_class import FoodRecipeDataset, ApplyTransforms
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


def prepare_dataloaders(batch_size, pretrained=False, image_size=(224,224), augment=False):
    
    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer("basic_english")
    transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])

    if pretrained:
        if augment:
            VocabImage = FoodRecipeDataset(text_path, images_path, transform=None)
        else:
            VocabImage = FoodRecipeDataset(text_path, images_path, transform=transform)
    else:
        VocabImage = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=transform)
    
    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImage))
    test_size = len(VocabImage) - training_size
    generator = torch.Generator().manual_seed(42)
    training_data, temp = random_split(VocabImage, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)
    
    if augment:
        training_data = ApplyTransforms(training_data, split='train')
        val_data = ApplyTransforms(val_data, split='val')
        test_data = ApplyTransforms(test_data, split='test')

    if pretrained:
        trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
    else:
        trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return trainloader, valloader, testloader, training_data, val_data, test_data

def dump_pickles(features1, features2, ids, pkl_file_1="models/test_img_features.pkl", pkl_file_2="models/test_text_features.pkl"):
    with open(pkl_file_1, 'wb') as f:
        pickle.dump((features1, ids), f)
    with open(pkl_file_2, 'wb') as f:
        pickle.dump((features2, ids), f)


def inference(model_path="models/best_model_ever.pt", pkl_file_1="models/test_img_features.pkl", pkl_file_2="models/test_img_features.pkl", pretrained=False, augment=False):

    model = torch.load(model_path)
    model.eval()

    _, _, testloader, _, _, _ = prepare_dataloaders(batch_size=1, pretrained=True, image_size=(224,224), augment=augment)


    test_img_features = []
    test_text_features = []
    ids = []

    for image, title, ingredients, instructions, cleaned_ingredients, id in tqdm.tqdm(testloader):
        if torch.cuda.is_available():
                if pretrained:
                    image = image.to('cuda')
                else:
                    image, title, ingredients, instructions, cleaned_ingredients = image.to('cuda'), title.to('cuda'), ingredients.to('cuda'), instructions.to('cuda'), cleaned_ingredients.to('cuda')

        ids.append(id.item())
        image_features, text_features = model(image, title, ingredients, instructions)

        test_img_features.append(image_features.detach().cpu().numpy()[0])
        test_text_features.append(text_features.detach().cpu().numpy()[0])


    dump_pickles(features1=np.asarray(test_img_features), features2=np.asarray(test_text_features), ids=np.array(ids), pkl_file_1=pkl_file_1, pkl_file_2=pkl_file_2)

if __name__ == "__main__":
    models = os.listdir("models/")
    augment = False
    for model in models:
        if model.endswith(".pt"):
            print("evaluating:", model)
            name = model.split("_")[-1].split(".")[0]
            if name not in ["1", "2", "3"]:
                augment = True
            # print(augment)
            inference(pretrained=True, model_path=f"models/{model}", pkl_file_1=f"models/features/test_img_features_{name}.pkl", pkl_file_2=f"models/features/test_text_features_{name}.pkl", augment=augment)
        # inference(pretrained=True, model_path=f"models/best_model_ever_{i}.pt", pkl_file_1=f"models/test_img_features_{i}.pkl", pkl_file_2=f"models/test_text_features_{i}.pkl")