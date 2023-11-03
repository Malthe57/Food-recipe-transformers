from image_encoder import ImageEncoder
from text_encoder import TextEncoder
import torch.nn as nn
import torch

class JointEncoder(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=128):

        super().__init__()        
        self.image_encoder = image_encoder

        self.text_encoder = text_encoder

        # linear layer to merge features from all recipe components.
        self.text_linear = nn.Linear(embed_dim*3, embed_dim)
        self.img_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, img, title, ingredients, instructions):

        img_feat = self.img_linear(self.image_encoder(img))

        title_feat = self.text_encoder(title)  
        ingredients_feat = self.text_encoder(ingredients)
        instructions_feat = self.text_encoder(instructions)

        text_features = torch.cat([title_feat, ingredients_feat, instructions_feat], dim=1)
        self.merger(text_features)

        return img_feat, text_features

