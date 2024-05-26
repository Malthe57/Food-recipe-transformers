import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
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
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import glob


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

def plot_examples(metrics_img2text, metrics_text2img, VocabImage, ids):

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

def plot_topfive_img2text(metrics_img2text, VocabImage, ids):

    n_examples = 0

    for i, (idx, pred_idx) in enumerate(zip(ids,metrics_img2text['pred_idx'])):

        if idx == pred_idx:
            n_examples += 1
            fig, ax = plt.subplots(1,6, figsize=(10,5))
            fig.subplots_adjust(hspace=0.2, wspace=0.1)
            
            ax = ax.ravel()
            print(idx)

            input = VocabImage[idx]
            pred = VocabImage[pred_idx]

            image = input[0]
            invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                    ])

            image = invTrans(image)

            ax[0].imshow(image.permute(1,2,0))
            ax[0].set_title("Input: \n" + input[1])
            ax[0].axis('off')
            ax[0].title.set_fontsize(ax[0].title.get_fontsize()-5)

            _, w, h = image.shape

            top5_preds = metrics_img2text['top5_pred_idx'][i]
            for j in range(len(top5_preds)):
                pred = VocabImage[top5_preds[j]]
                pred_text = pred[1] + pred[2] + pred[3]
                wordcloud = WordCloud(width=w, height=h).generate(pred_text)
                ax[j+1].imshow(wordcloud, interpolation = 'bilinear')
                ax[j+1].set_title(f"Prediction {j+1}: \n" + pred[1])
                ax[j+1].axis('off')
                ax[j+1].title.set_fontsize(ax[j+1].title.get_fontsize()-5)
            plt.tight_layout()
            plt.show()

            if n_examples >= 100:
                break

def plot_topfive_text2img(metrics_text2img, VocabImage, ids):

    n_examples = 0

    for i, (idx, pred_idx) in enumerate(zip(ids,metrics_text2img['pred_idx'])):

        if idx == pred_idx:

            n_examples += 1

            fig, ax = plt.subplots(1,6, figsize=(10,5))
            fig.subplots_adjust(hspace=0.2, wspace=0.1)
            
            ax = ax.ravel()
            print(idx)

            input = VocabImage[idx]
            
            invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                    ])
            true_image = invTrans(input[0])
            text = input[1] + input[2] + input[3]
            
            wordcloud = WordCloud(width=224, height=224).generate(text)
            ax[0].imshow(wordcloud, interpolation = 'bilinear')
            ax[0].set_title("Input: \n" + input[1])
            ax[0].axis('off')
            ax[0].title.set_fontsize(ax[0].title.get_fontsize()-5)

            for j in range(5):
                pred = VocabImage[metrics_text2img['top5_pred_idx'][i][j]]
                pred_image = invTrans(pred[0])
                ax[j+1].imshow(pred_image.permute(1,2,0))
                ax[j+1].set_title(f"Prediction {j+1}: \n" + pred[1])
                ax[j+1].axis('off')
                ax[j+1].title.set_fontsize(ax[j+1].title.get_fontsize()-5)
            plt.tight_layout()
            plt.show()

            if n_examples >= 100:
                break

def animate(mode='img2text'):
    if mode == 'img2text':
        files = glob.glob('reports/figures/img2text/*.png')
    elif mode == 'text2img':
        files = glob.glob('reports/figures/text2img/*.png')

    # Create the figure and axes objects
    fig, ax = plt.subplots()
    fig.set_size_inches(5.6*4,1*4)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    image_array = []  
    for file in files:
        image = np.array(Image.open(file))
        img = ax.imshow(image, animated=True)
        image_array.append([img])   

    # Create the animation object
    animation_fig = ArtistAnimation(fig, image_array, interval=1500, blit=True, repeat_delay=50)

    animation_fig.save(f"reports/figures/{mode}.gif")


if __name__ == '__main__':

    _, _, _, _, _, test_data, VocabImage = prepare_dataloaders(batch_size=1, pretrained=True, image_size=(224,224))

    img_features, ids = pickle.load(open(f"models/features/test_img_features_lr3.pkl", 'rb'))
    text_features, ids = pickle.load(open(f"models/features/test_text_features_lr3.pkl", 'rb'))

    metrics_img2text = compute_metrics(img_features, text_features, ids, metric='cosine', recall_klist=(1, 5, 10), return_raw=False, return_idx=True)
    metrics_text2img = compute_metrics(text_features, img_features, ids, metric='cosine', recall_klist=(1, 5, 10), return_raw=False, return_idx=True)

    # plot_examples(metrics_img2text, metrics_text2img, VocabImage, ids)
    # plot_topfive_img2text(metrics_img2text, VocabImage, ids)
    # plot_topfive_text2img(metrics_text2img, VocabImage, ids)

    animate('img2text')
    animate('text2img')

