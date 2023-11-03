import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from VocabImagedataset_class import VocabImageDataset, pad_input, collate_batch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
from torchtext.data.utils import get_tokenizer

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        ####################### insert code here ####################### 
        self.k_projection = nn.Linear(embed_dim, num_heads * self.head_dim )
        self.q_projection = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.v_projection = nn.Linear(embed_dim, num_heads * self.head_dim)
        ################################################################
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_length, embed_dim = x.size()
        keys    = self.k_projection(x)
        queries = self.q_projection(x)
        values  = self.v_projection(x)

        # Rearrange keys, queries and values 
        # from batch_size x seq_length x embed_dim to (batch_size x num_head) x seq_length x head_dim
        keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        values = rearrange(values, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)

        ####################### insert code here ####################### 
        score = torch.matmul(queries, torch.permute(keys, (0,2,1))) / math.sqrt(embed_dim)
        attention = F.softmax(score, dim=0)
        out = torch.matmul(attention, values)
        ################################################################

        # Rearragne output
        # from (batch_size x num_head) x seq_length x head_dim to batch_size x seq_length x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0., max_seq_len).unsqueeze(1)

        # Your implemntation
        ####################### insert code here #######################

        # for p in range(max_seq_len):
        #     for i in range(embed_dim):
        #         if i % 2 == 0:
        #             pe[p,i] = torch.sin(position[p] * (1/(10000**(2*i / embed_dim))))
        #         else:
        #             pe[p,i] = torch.cos(position[p] * (1/(10000**(2*i / embed_dim))))

        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        ################################################################

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]
        #return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEmbedding, self).__init__()
        # Your implemntation
        ####################### insert code here ####################### 
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        ################################################################

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        # Your implemntation
        ####################### insert code here ####################### 
        x = x + self.positional_embedding(torch.arange(seq_length).to(x.device))
        return x
        ################################################################

class TextEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.0, 
                fc_dim=None, num_tokens=50_000, num_classes=2, pool='mean', pos_enc='learnable'
    ):
        super().__init__()

        assert pool in ['mean', 'max']
        assert pos_enc in ['fixed', 'learnable']
        
        self.pool, self.pos_enc, = pool, pos_enc


        self.token_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=num_tokens)
        
        if self.pos_enc == 'learnable':
            self.positional_encoding = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        elif self.pos_enc == 'fixed':
            self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)

        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        tokens = self.token_embedding(x)
        batch_size, seq_length, embed_dim = tokens.size()

        # positional encoding
        # learnable: torch.Size([32, 12, 128])


        if self.pos_enc == 'fixed':
            x = self.positional_encoding(tokens) # torch.Size([32, 9, 128])
        elif self.pos_enc == 'learnable':
            x = tokens + self.positional_encoding.to(tokens.device, dtype=tokens.dtype)(tokens)
        x = self.dropout(x) 
        x = self.transformer_blocks(x)

        if self.pool =='max':
            x = x.max(dim=1)[0]
        elif self.pool =='mean':
            x = x.mean(dim=1)
        
        return x
 
if __name__ == '__main__':
    title_encoder = TextEncoder(embed_dim=128, num_heads=4, num_layers=4, max_seq_len=512, dropout=0.0, fc_dim=None, num_tokens=50_000, num_classes=2, pool='mean', pos_enc='learnable')
    ingredients_encoder = TextEncoder(embed_dim=128, num_heads=4, num_layers=4, max_seq_len=512, dropout=0.0, fc_dim=None, num_tokens=50_000, num_classes=2, pool='mean', pos_enc='learnable')
    instructions_encoder = TextEncoder(embed_dim=128, num_heads=4, num_layers=4, max_seq_len=512, dropout=0.0, fc_dim=None, num_tokens=50_000, num_classes=2, pool='mean', pos_enc='learnable')

    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer('basic_english')

    image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor() ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VocabImage = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=image_transform)
    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImage))
    test_size = len(VocabImage) - training_size
    generator = torch.Generator().manual_seed(42)
    train_data, temp = random_split(VocabImage, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)
    
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)

    for image, title, ingredients, instructions, cleaned_ingredients in train_dataloader:
        title_output = title_encoder(title)
        ingredients_output = ingredients_encoder(ingredients)
        instructions_output = instructions_encoder(instructions)

        output = torch.cat((title_output, ingredients_output, instructions_output), dim=1)
        print("Text transformer output shape:", output.shape)
        break
