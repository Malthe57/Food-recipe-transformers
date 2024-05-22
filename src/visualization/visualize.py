import matplotlib.pyplot as plt
import pickle
import sys
import os
sys.path.append("../Food-recipe-transformers/src/")
from utils.metrics import compute_metrics
from dataset_class import FoodRecipeDataset
from VocabImagedataset_class import VocabImageDataset, pad_input, collate_batch
from wordcloud import WordCloud
import torch
from torchtext.data.utils import get_tokenizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from numpy.random import rand


def prepare_dataloaders(batch_size, pretrained=False, image_size=(64,64)):
    
    current_working_directory = os.getcwd()
    images_path = os.path.join(current_working_directory, "src/dataset/Food Images")
    text_path = os.path.join(current_working_directory, "src/dataset/food.csv")
    vocab = torch.load("src/dataset/vocab.pt")
    tokenizer = get_tokenizer("basic_english")
    transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pretrained:
        VocabImage = FoodRecipeDataset(text_path, images_path, transform=transform)
    else:
        VocabImage = VocabImageDataset(annotations_file=text_path, img_dir=images_path, vocab=vocab, tokenizer=tokenizer, device=device, transform=transform)
    
    training_share = 0.8 #Proportion of data that is alotted to the training set
    training_size = int(training_share*len(VocabImage))
    test_size = len(VocabImage) - training_size
    generator = torch.Generator().manual_seed(42)
    training_data, temp = random_split(VocabImage, [training_size, test_size], generator)
    test_data, val_data = random_split(temp, [int(0.4*len(temp)), int(0.6*len(temp))], generator)

    if pretrained:
        trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
    else:
        trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return trainloader, valloader, testloader, training_data, val_data, test_data, VocabImage

_, _, _, _, _, test_data, VocabImage = prepare_dataloaders(batch_size=1, pretrained=True, image_size=(224,224))

img_features, ids = pickle.load(open(f"models/features/test_img_features_lr3.pkl", 'rb'))
text_features, ids = pickle.load(open(f"models/features/test_text_features_lr3.pkl", 'rb'))

metrics_img2text = compute_metrics(img_features, text_features, ids, metric='cosine', recall_klist=(1, 5, 10), return_raw=False, return_idx=True)
metrics_text2img = compute_metrics(text_features, img_features, ids, metric='cosine', recall_klist=(1, 5, 10), return_raw=False, return_idx=True)



n_images = 4
fig, ax = plt.subplots(n_images,3, figsize=(10,10))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
i=1
for (idx, pred_idx) in zip(ids,metrics_img2text['pred_idx']):

    if rand(1) < 0.95:
        None
    else:

        input = VocabImage[idx]
        pred = VocabImage[pred_idx]

        image = input[0]
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

        image = invTrans(image)
        pred_text = pred[1] + pred[2] + pred[3]
        true_text = input[1] + input[2] + input[3]
        ax[i,0].imshow(image.permute(1,2,0))
        ax[i,0].set_title("Input: \n" + input[1])
        ax[i,0].axis('off')
        ax[i,0].title.set_fontsize(ax[i,0].title.get_fontsize()-5)
        wordcloud = WordCloud().generate(pred_text)
        wordcloud_true = WordCloud().generate(true_text)
        ax[i,1].imshow(wordcloud, interpolation = 'bilinear')
        ax[i,1].set_title("Prediction: \n" + pred[1])
        ax[i,1].axis('off')
        ax[i,1].title.set_fontsize(ax[i,1].title.get_fontsize()-5)


        ax[i,2].imshow(wordcloud_true, interpolation = 'bilinear')
        ax[i,2].set_title("True: \n" + input[1])
        ax[i,2].axis('off')
        ax[i,2].title.set_fontsize(ax[i,2].title.get_fontsize()-5)

        i += 1
        if i >= (n_images):
            break
i=0
input = VocabImage[metrics_img2text['idx'][4]]
pred = VocabImage[metrics_img2text['idx'][4]]

image = input[0]
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                std = [ 1., 1., 1. ]),
                        ])

image = invTrans(image)
pred_text = pred[1] + pred[2] + pred[3]
true_text = input[1] + input[2] + input[3]
ax[i,0].imshow(image.permute(1,2,0))
ax[i,0].set_title("Input: \n" + input[1])
ax[i,0].axis('off')
ax[i,0].title.set_fontsize(ax[i,0].title.get_fontsize()-5)
wordcloud = WordCloud().generate(pred_text)
wordcloud_true = WordCloud().generate(true_text)
ax[i,1].imshow(wordcloud, interpolation = 'bilinear')
ax[i,1].set_title("Prediction: \n" + pred[1])
ax[i,1].axis('off')
ax[i,1].title.set_fontsize(ax[i,1].title.get_fontsize()-5)


ax[i,2].imshow(wordcloud_true, interpolation = 'bilinear')
ax[i,2].set_title("True: \n" + input[1])
ax[i,2].axis('off')
ax[i,2].title.set_fontsize(ax[i,2].title.get_fontsize()-5)
# plt.savefig("models/sample_results_w_wordclouds_img2txt_mar1")
plt.show()


fig, ax = plt.subplots(n_images,3, figsize=(10,10))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
i=1
for (idx, pred_idx) in zip(ids,metrics_text2img['pred_idx']):

    if rand(1) < 0.95:
        None
    else:

        input = VocabImage[idx]
        pred = VocabImage[pred_idx]

        
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])
        true_image = invTrans(input[0])
        pred_image = invTrans(pred[0])
        
        text = input[1] + input[2] + input[3]
        
        wordcloud = WordCloud().generate(text)
        ax[i,0].imshow(wordcloud, interpolation = 'bilinear')
        ax[i,0].set_title("Input: \n" + input[1])
        ax[i,0].axis('off')
        ax[i,0].title.set_fontsize(ax[i,0].title.get_fontsize()-5)

        ax[i,1].imshow(pred_image.permute(1,2,0))
        ax[i,1].set_title("Prediction: \n" + pred[1])
        ax[i,1].axis('off')
        ax[i,1].title.set_fontsize(ax[i,1].title.get_fontsize()-5)

        ax[i,2].imshow(true_image.permute(1,2,0))
        ax[i,2].set_title("True: \n" + input[1])
        ax[i,2].axis('off')
        ax[i,2].title.set_fontsize(ax[i,2].title.get_fontsize()-5)

        i += 1
        if i >= (n_images):
            break

input = VocabImage[metrics_text2img['idx'][0]]
pred = VocabImage[metrics_text2img['idx'][0]]
i=0

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                std = [ 1., 1., 1. ]),
                        ])
true_image = invTrans(input[0])
pred_image = invTrans(pred[0])

text = input[1] + input[2] + input[3]

wordcloud = WordCloud().generate(text)
ax[i,0].imshow(wordcloud, interpolation = 'bilinear')
ax[i,0].set_title("Input: \n" + input[1])
ax[i,0].axis('off')
ax[i,0].title.set_fontsize(ax[i,0].title.get_fontsize()-5)

ax[i,1].imshow(pred_image.permute(1,2,0))
ax[i,1].set_title("Prediction: \n" + pred[1])
ax[i,1].axis('off')
ax[i,1].title.set_fontsize(ax[i,1].title.get_fontsize()-5)

ax[i,2].imshow(true_image.permute(1,2,0))
ax[i,2].set_title("True: \n" + input[1])
ax[i,2].axis('off')
ax[i,2].title.set_fontsize(ax[i,2].title.get_fontsize()-5)
# plt.savefig("models/sample_results_w_wordclouds_txt2img_mar1")
plt.show()

