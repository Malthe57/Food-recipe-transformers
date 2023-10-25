# Imports
import os
import pandas as pd

import torch
from PIL import Image
from torch.utils.data import Dataset


class FoodRecipeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_text = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_text)

    def __getitem__(self, idx):
        img_path = str(os.path.join(self.img_dir, self.img_text.iloc[idx, 3]) + ".jpg")
        image = Image.open(img_path)
        title = str(self.img_text.iloc[idx, 0])
        ingredients = str(self.img_text.iloc[idx, 1])
        instructions = str(self.img_text.iloc[idx, 2])
        cleaned_ingredients = str(self.img_text.iloc[idx, 4])




        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            title = self.target_transform(title)
            ingredients = self.target_transform(ingredients)
            instructions = self.target_transform(instructions)
            cleaned_ingredients = self.target_transform(cleaned_ingredients)


        return image, title, ingredients, instructions, cleaned_ingredients

