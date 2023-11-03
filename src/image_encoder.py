import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from VocabImagedataset_class import VocabImageDataset, pad_input, collate_batch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
from torchtext.data.utils import get_tokenizer

def positional_encoding_2d(nph, npw, dim, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(nph), torch.arange(npw), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.k_projection  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_projeciton  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_len, embed_dim = x.size()
        keys    = self.k_projection(x)
        queries = self.q_projection(x)
        values  = self.v_projeciton(x)

        # Rearrange keys, queries and values 
        # from batch_size x seq_len x embed_dim to (batch_size x num_head) x seq_len x head_dim
        keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        values = rearrange(values, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)

        attention_logits = torch.matmul(queries, keys.transpose(1, 2))
        attention_logits = attention_logits * self.scale
        attention = F.softmax(attention_logits, dim=-1)
        self.attention_vals = attention
        out = torch.matmul(attention, values)

        # Rearragne output
        # from (batch_size x num_head) x seq_len x head_dim to batch_size x seq_len x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_len, seq_len)
        assert out.size() == (batch_size, seq_len, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.GELU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layernorm1(attention_out + x)
        x = self.dropout(x)
        fc_out = self.fc(x)
        x = self.layernorm2(fc_out + x)
        x = self.dropout(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, image_size, channels, patch_size, embed_dim, num_heads, num_layers,
                 pos_enc='learnable', pool='cls', dropout=0.0, 
                 fc_dim=None, num_classes=2, ):
        
        super().__init__()

        assert pool in ['cls', 'mean', 'max']
        assert pos_enc in ['fixed', 'learnable']

        self.pool, self.pos_enc, = pool, pos_enc

        H, W = image_size
        patch_h, patch_w = patch_size
        assert H % patch_h == 0 and W % patch_w == 0, 'Image dimensions must be divisible by the patch size'

        num_patches = (H // patch_h) * (W // patch_w)
        patch_dim = channels * patch_h * patch_w

        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.rand(1,1,embed_dim))
            num_patches += 1
        
        # TASK: Implement patch embedding layer 
        #       Convert images to patches and project to the embedding dimension
        # HINT: 1) Use the Rearrange layer from einops.layers.torch 
        #          in the same way you used the rearrange function 
        #          in the image_to_patches function (playground.py)
        #       2) Stack Rearrange layer with a linear projection layer using nn.Sequential
        #          Consider including LayerNorm layers before and after the linear projection
        ######## insert code here ########
        #
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (nph ph) (npw pw) -> b (nph npw) (ph pw c)', ph=patch_h, pw=patch_w),
                      nn.LayerNorm(patch_dim),
                      nn.Linear(patch_dim, embed_dim),
                      nn.LayerNorm(embed_dim))
        #
        #
        #################################

        if self.pos_enc == 'learnable':
            self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        elif self.pos_enc == 'fixed':
            self.positional_embedding = positional_encoding_2d(
                nph = H // patch_h, 
                npw = W // patch_w,
                dim = embed_dim,
            )  

        transformer_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)


    def forward(self, img):

        tokens = self.to_patch_embedding(img)
        batch_size, num_patches, embed_dim = tokens.size()
        
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=batch_size)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            num_patches+=1
        
        # fixed: # torch.Size([32, 9, 128])
        # learnable: # torch.Size([1, 65, 128])

        positions =  self.positional_embedding.to(img.device, dtype=img.dtype)
        if self.pos_enc == 'fixed' and self.pool=='cls':
            positions = torch.cat([torch.zeros(1, embed_dim).to(img.device), positions], dim=0)
        x = tokens + positions
        
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        
        if self.pool =='max':
            x = x.max(dim=1)[0]
        elif self.pool =='mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[:, 0]

        return x
    
    
if __name__ == '__main__':
    print('Testing ImageEncoder')
    image_size = (64, 64)
    patch_size = (8,8)
    channels = 3
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    pos_enc = 'learnable'
    pool = 'cls'

    encoder = ImageEncoder(image_size=image_size, channels=channels, patch_size=patch_size, embed_dim=128, num_heads=4, num_layers=4, pos_enc='learnable', pool='cls')

    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer('basic_english')

    image_transform = transforms.Compose([
    transforms.Resize(image_size),  
    transforms.ToTensor()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VocabImage = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=image_transform)
    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImage))
    test_size = len(VocabImage) - training_size
    generator = torch.Generator().manual_seed(42)
    train_data, temp = random_split(VocabImage, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)
    
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)

    encoder.to(device=device)

    for image, title, ingredients, instructions, cleaned_ingredients in train_dataloader:
        output = encoder(image)

        print("Image transformer output shape:", output.shape)

        break