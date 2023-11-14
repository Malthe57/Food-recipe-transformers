import torch
import clip
from PIL import Image
from VocabImagedataset_class import VocabImageDataset, pad_input, collate_batch, collate_clip
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
import os
from torch.utils.data import DataLoader, random_split

def prepare_dataloaders(batch_size):
    
    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer("basic_english")
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VocabImageData = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=transform, vocabularise=False)


    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImageData))
    test_size = len(VocabImageData) - training_size
    generator = torch.Generator().manual_seed(42)
    training_data, temp = random_split(VocabImageData, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)

    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_clip)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    return trainloader, valloader, testloader, training_data, val_data, test_data

trainloader, valloader, testloader, training_data, val_data, test_data = prepare_dataloaders(batch_size=32)

for image, title, ingredients, instructions, cleaned_ingredients, _ in trainloader:
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("src/dataset/Food Images/3-ingredient-fudge-pops.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["an icecream", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 
