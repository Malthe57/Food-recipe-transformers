import torch
import clip
from PIL import Image
from CLIPDataset_class import ClipFoodRecipeDataset
from VocabImagedataset_class import VocabImageDataset, pad_input, collate_batch, collate_clip
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
import os
from torch.utils.data import DataLoader, random_split
import numpy as np  
import pickle


current_working_directory = os.getcwd()
images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
text_path = os.path.join(current_working_directory, "src/dataset/food.csv")

VocabImage = ClipFoodRecipeDataset(text_path, images_path, transform=None)

training_share = 0.8 #Proportion of data that is alotted to the training set
training_size = int(training_share*len(VocabImage))
test_size = len(VocabImage) - training_size
generator = torch.Generator().manual_seed(42)
training_data, temp = random_split(VocabImage, [training_size, test_size], generator)
test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)

def collate_clip(batch):

    image, title, ingredients, instructions, cleaned_ingredients, id = zip(*batch)

    image_tensor = [img for img in [transforms.ToPILImage(mode='RGB')(img) for img in image]]
    title_tensor = title
    ingredients_tensor = ingredients
    instructions_tensor = instructions
    cleaned_ingredients_tensor = cleaned_ingredients

    return image_tensor, title_tensor, ingredients_tensor, instructions_tensor, cleaned_ingredients_tensor, id[0] # get id of tuple

def dump_pickles(features1, features2, ids, pkl_file_1="models/test_img_features.pkl", pkl_file_2="models/test_text_features.pkl"):
    with open(pkl_file_1, 'wb') as f:
        pickle.dump((features1, ids), f)
    with open(pkl_file_2, 'wb') as f:
        pickle.dump((features2, ids), f)

def inference(pkl_file_1="models/features/test_img_features.pkl", pkl_file_2="models/features/test_img_features.pkl"):

    testloader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_clip)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)


    test_img_features = []
    test_text_features = []
    ids = []

    for image, title, ingredients, instructions, cleaned_ingredients, idx in testloader:

        image_input =  torch.stack([preprocess(img) for img in image])
        text_inputs = clip.tokenize(["".join(t) for t in title])
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            ids.append(idx)

            test_img_features.append(image_features.detach().cpu().numpy()[0])
            test_text_features.append(text_features.detach().cpu().numpy()[0])

            dump_pickles(features1=np.asarray(test_img_features), features2=np.asarray(test_text_features), ids=np.array(ids), pkl_file_1=pkl_file_1, pkl_file_2=pkl_file_2)

if __name__ == "__main__":
    inference(pkl_file_1=f"models/features/test_img_features_clip.pkl", pkl_file_2=f"models/features/test_text_features_clip.pkl")