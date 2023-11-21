# Imports
import os
import pandas as pd

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split



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
        rgb_im = image.convert('RGB')
        title = str(self.img_text.iloc[idx, 0])
        ingredients = str(self.img_text.iloc[idx, 1])
        instructions = str(self.img_text.iloc[idx, 2])
        cleaned_ingredients = str(self.img_text.iloc[idx, 4])


        if self.transform:
            rgb_im = self.transform(rgb_im)
        if self.target_transform:
            title = self.target_transform(title)
            ingredients = self.target_transform(ingredients)
            instructions = self.target_transform(instructions)
            cleaned_ingredients = self.target_transform(cleaned_ingredients)


        return rgb_im, title[:512], ingredients[:512], instructions[:512], cleaned_ingredients[:512], idx
    
class ApplyTransforms(Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
        self.train_transforms = transforms.Compose([transforms.Resize((256,256)), 
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomCrop(224),
                                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])
        
        self.test_transforms = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rgb_im, title, ingredients, instructions, cleaned_ingredients, idx = self.dataset[idx]
        if self.split == 'train':
            rgb_im = self.train_transforms(rgb_im)
            # print("Applying train transforms: RandomCrop")
        elif self.split == 'val':
            rgb_im = self.test_transforms(rgb_im)
            # print("Applying val transforms: CenterCrop")
        elif self.split == 'test':
            rgb_im = self.test_transforms(rgb_im)
            # print("Applying test transforms: CenterCrop")

        return rgb_im, title, ingredients, instructions, cleaned_ingredients, idx


if __name__ == "__main__":
    batch_size = 32

    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")

    VocabImage = FoodRecipeDataset(text_path, images_path, transform=None)

    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImage))
    test_size = len(VocabImage) - training_size
    generator = torch.Generator().manual_seed(42)
    training_data, temp = random_split(VocabImage, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)

    training_data = ApplyTransforms(training_data, split='train')
    val_data = ApplyTransforms(val_data, split='val')
    test_data = ApplyTransforms(test_data, split='test')

    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)