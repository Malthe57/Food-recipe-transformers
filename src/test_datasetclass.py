
import os
import torch
from torch.utils.data import DataLoader, random_split
from dataset_class import FoodRecipeDataset
from torchvision import transforms
import matplotlib.pyplot as plt

current_working_directory = os.getcwd()
images_path = os.path.join(current_working_directory, "dataset/Food Images")
text_path = os.path.join(current_working_directory, "dataset/food.csv")

FoodRecipeData = FoodRecipeDataset(text_path, images_path, transforms.ToTensor())


training_share = 0.6 #Proportion of data that is alotted to the training set
training_size = int(training_share*len(FoodRecipeData))
test_size = len(FoodRecipeData) - training_size
generator = torch.Generator().manual_seed(42)
training_data, test_data = random_split(FoodRecipeData, [training_size, test_size], generator)

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

image, title, ingredients, instructions, cleaned_ingredients = next(iter(train_dataloader))
plt.imshow(image[0].permute(1,2,0))
plt.show()
