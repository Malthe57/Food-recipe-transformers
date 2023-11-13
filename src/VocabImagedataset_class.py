import pandas as pd
import torch
from  PIL import Image
import os
import sys
sys.path.append('../src/')
from dataset_class import FoodRecipeDataset


class VocabImageDataset(FoodRecipeDataset):
    def __init__(self, annotations_file, img_dir, vocab, tokenizer, max_seq_len=512, device='cpu', transform=None):
        self.recipe_text = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.transform = transform
        
        self.text_pipeline = lambda x : vocab(tokenizer(x))
        
    def __len__(self):
        return len(self.recipe_text)
        
    def __getitem__(self, idx):
        img_path = str(os.path.join(self.img_dir, self.recipe_text.iloc[idx, 3]) + ".jpg")
        image = Image.open(img_path)
        rgb_im = image.convert('RGB')
        title_seq = str(self.recipe_text.iloc[idx, 0])
        ingredients_seq = str(self.recipe_text.iloc[idx, 1])
        instructions_seq = str(self.recipe_text.iloc[idx, 2])
        cleaned_ingredients_seq = str(self.recipe_text.iloc[idx, 4])
        
        title = torch.tensor(self.text_pipeline(title_seq))[:self.max_seq_len]
        ingredients = torch.tensor(self.text_pipeline(ingredients_seq))[:self.max_seq_len]
        instructions = torch.tensor(self.text_pipeline(instructions_seq))[:self.max_seq_len]
        cleaned_ingredients = torch.tensor(self.text_pipeline(cleaned_ingredients_seq))[:self.max_seq_len]
        
        output_dict = {'image': self.transform(rgb_im).to(self.device), 'title': title.to(self.device), 'ingredients': ingredients.to(self.device),
                       'instructions': instructions.to(self.device), 'cleaned_ingredients': cleaned_ingredients.to(self.device)}
        
#         return output_dict
        return self.transform(rgb_im).to(self.device), title.to(self.device), ingredients.to(self.device), instructions.to(self.device), cleaned_ingredients.to(self.device), idx

def pad_input(input,):
    """
    creates a padded tensor to fit the longest sequence in the batch
    
    Adapted from: https://github.com/amzn/image-to-recipe-transformers/blob/96e257e910c79a5411c3f65f598dd818f72fc262/src/dataset.py#L123C1-L141C19
    """
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        
        # get idx token corresponding to <pad> in vocab, aka 1
        # pad_token = vocab.get_stoi()['<pad>']
        pad_token = 1
        
        targets = torch.zeros(len(input), max(l)).long()+pad_token
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
            
        # get idx token corresponding to <pad> in vocab
        # pad_token = vocab.get_stoi()['<pad>']
        pad_token = 1
        
        
        targets = torch.zeros(len(input), max(n), max(l)).long()+pad_token
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_batch(batch):
    
    image, title, ingredients, instructions, cleaned_ingredients, id = zip(*batch)
    
    image_tensor = torch.stack(image, dim=0)
    title_tensor = pad_input(title)
    ingredients_tensor = pad_input(ingredients)
    instructions_tensor = pad_input(instructions)
    cleaned_ingredients_tensor = pad_input(cleaned_ingredients)
    
    return image_tensor, title_tensor, ingredients_tensor, instructions_tensor, cleaned_ingredients_tensor, id[0] # get id of tuple
    
    
    
    