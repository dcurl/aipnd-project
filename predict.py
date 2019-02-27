#Imports
import json
import sys

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

import argparse

#Create argparse for easy user entry
parser = parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type = str, default = 'flowers/test/52/image_04200.jpg', help = 'Directory of Dataset')
parser.add_argument('--load_chkpt', type = str, default = 'checkpoint.pth', help='Load the trained model from a file')
parser.add_argument('--gpu', type = bool, default = 'True', help='True for GPU, False for CPU')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help='File containing classifier categories')
parser.add_argument('--top_k', type = int, default = 5, help = 'Number of Top Categories to return')
args = parser.parse_args()

# Set variables based on user entry
if args.img_path is not None:
    img_path = args.img_path

if args.gpu is not None:
    gpu = args.gpu

if args.load_chkpt is not None:
    load_chkpt = args.load_chkpt

if args.category_names is not None:
    category_names = args.category_names

if args.top_k is not None:
    top_k = args.top_k


# Load checkpoint Function
def load_checkpoint(filepath):
    checkpoint= torch.load(filepath)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Process Image Function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # Set Variables
    means = [0.485, 0.456, 0.406]
    std_devs = [0.229, 0.224, 0.225]

    # Processing Steps
    process_img = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, std_devs)])

    # Process image through Processing Steps
    pil_img = Image.open(image)
    pil_img = process_img(pil_img).float()
    np_img = np.array(pil_img)

    return np_img

# Display Image Function
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# Prediction function
def predict(image_path, model, topk=5):
    # Process image
    sample_img = process_image(image_path)

    # Convert Numpy to Tensor
    img_tensor = torch.from_numpy(sample_img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    sample_input = img_tensor.unsqueeze(0)

    # Get Probabilities
    ps = torch.exp(model.forward(sample_input))
    ps, labels = torch.topk(ps, topk)

    # Seperate Probabilities (ps), Numeric Labels (labels) into individual lists
    top_ps = ps.detach().numpy().tolist()[0]
    top_labels = labels.detach().numpy().tolist()[0]

    # Gather Flower Labels and convert Numeric Label to Flower Label
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_flowers = [cat_to_name[idx_to_class[label]] for label in top_labels]


    # Return Probabilities (ps), Numeric Labels (Labels), Flower Labels
    return top_ps, top_labels, top_flowers


# Test Loading Checkpoint
#Load model from checkpoint saved in train.py
model = load_checkpoint(load_chkpt)

#Switch to CPU and Evaluation
model.to('cpu')
model.eval()

# Set Test Image
image_path = img_path

# Make prediction
ps, labels, flowers = predict(image_path, model, top_k)

# Print Prediction
print("Image: ", img_path)
print("\nTop ", top_k, " Predictions")
print("Percentage: ", ps)
print("Numeric Label: ", labels)
print("Flower Label: ", flowers)
print("\nTop Prediction")
print("Percentage: ", round(ps[0] * 100, 2), "%")
print("Numeric Label: ", labels[0])
print("Flower Label: ", flowers[0])
