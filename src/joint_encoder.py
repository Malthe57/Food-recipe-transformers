from image_encoder import ImageEncoder
from text_encoder import TextEncoder
import torch.nn as nn
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from VocabImagedataset_class import VocabImageDataset, pad_input, collate_batch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
from torchtext.data.utils import get_tokenizer

class JointEmbedding(nn.Module):
    def __init__(self, image_encoder, title_encoder, ingredients_encoder, instructions_encoder, embed_dim=128, only_title=False):

        super().__init__()        
        self.only_title = only_title
        self.image_encoder = image_encoder
        if only_title:
            self.title_encoder = title_encoder
        else:
            self.title_encoder = title_encoder
            self.ingredients_encoder = ingredients_encoder
            self.instructions_encoder = instructions_encoder

        # linear layer to merge features from all recipe components.
        if only_title:
            print("Training only on image and title")
            self.text_linear = nn.Linear(6*embed_dim, embed_dim)
            self.img_linear = nn.Linear(embed_dim, embed_dim)
        else:
            self.text_linear = nn.Linear(embed_dim*3, embed_dim)
            self.img_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, img, title, ingredients, instructions):

        if self.only_title:
            img_feat = self.img_linear(self.image_encoder(img))
            text_features = self.text_linear(self.title_encoder(title))

        else:
            img_feat = self.img_linear(self.image_encoder(img))

            title_feat = self.title_encoder(title)
            ingredients_feat = self.ingredients_encoder(ingredients)
            instructions_feat = self.instructions_encoder(instructions)

            text_features = self.text_linear(torch.cat([title_feat, ingredients_feat, instructions_feat], dim=1))

        return img_feat, text_features

if __name__ == '__main__':
    image_size = (64, 64)
    patch_size = (8,8)
    channels = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder = TextEncoder(embed_dim=128, num_heads=4, num_layers=4, max_seq_len=512, dropout=0.0, fc_dim=None, num_tokens=50_000, num_classes=2, pool='mean', pos_enc='learnable')
    text_encoder.to(device=device)
    image_encoder = ImageEncoder(image_size=image_size, channels=channels, patch_size=patch_size, embed_dim=128, num_heads=4, num_layers=4, pos_enc='learnable', pool='cls')
    image_encoder.to(device=device)
    joint_encoder = JointEmbedding(image_encoder, text_encoder, embed_dim=128)
    joint_encoder.to(device=device)



    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer('basic_english')

    image_transform = transforms.Compose([
    transforms.Resize((image_size)),  
    transforms.ToTensor() ])



    VocabImage = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=image_transform)
    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImage))
    test_size = len(VocabImage) - training_size
    generator = torch.Generator().manual_seed(42)
    train_data, temp = random_split(VocabImage, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)
    
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)

    for image, title, ingredients, instructions, cleaned_ingredients in train_dataloader:
        title, ingredients, instructions = title.to(device), ingredients.to(device), instructions.to(device)

        img_feat, text_feat = joint_encoder(image, title, ingredients, instructions)

        print("Image feature shape:", img_feat.shape)
        print("Text feature shape:", text_feat.shape)

        break
        
