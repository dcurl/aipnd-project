# Imports
import json
import sys

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse

#Create argparse for easy user entry
parser = parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'Directory of Dataset')
parser.add_argument('--gpu', type = bool, default = 'True', help='True for GPU, False for CPU')
parser.add_argument('--learn_rate', type = float, default = 0.001, help = 'Learning Rate (Ex: 0.001)')
parser.add_argument('--epochs', type = int, default = 10, help = 'Number of Epochs for Training')
parser.add_argument('--arch', type = str, default='densenet121', help='Architecture: Either densenet121 or vgg16')
parser.add_argument('--hidden_units', type = int, default = 256, help='Units for Hidden Layer')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help='Save the trained model to a file')
args = parser.parse_args()

# Set variables based on user entry
if args.data_dir is not None:
    data_directory = args.data_dir

if args.gpu is not None:
    gpu = args.gpu

if args.learn_rate is not None:
    learn_rate = args.learn_rate

if args.epochs is not None:
    epochs = args.epochs

if args.arch is not None:
    arch = args.arch

if args.hidden_units is not None:
    hidden_units = args.hidden_units

if args.save_dir is not None:
    save_dir = args.save_dir

# Function to set Directories for the Train, Validation, and Test Data
def load_data(data_directory = 'flowers'):
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return data_dir, train_dir, valid_dir, test_dir

# Function to transform data for Train, Validation, and Test Sets
def transform_data(data_dir, train_dir, valid_dir, test_dir):
    data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(data_dir, transform = data_transforms)
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets)
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    return train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader

# Function to build custom network and train it (calls the build_and_train Function)
def set_model(arch='densenet121', learn_rate = 0.001, hidden_units = [256], epochs=10, gpu = True):

    # TODO: Build and train your network
    # Load a pre-trained network, depending on user input
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
        model, model_classifier, model_optimizer, model_statedict, optimizer_statedict = build_and_train(model = model, learn_rate = learn_rate, input_size = input_size, hidden_size = hidden_units, epochs=epochs, gpu = gpu)

    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_size = 1024
        model, model_classifier, model_optimizer, model_statedict, optimizer_statedict = build_and_train(model = model, learn_rate = learn_rate, input_size = input_size, hidden_size = hidden_units, epochs=epochs, gpu = gpu)
    else:
        print('Please select Architecture as "vgg16" or "densenet121"')
        sys.exit()

    return model, input_size, model_classifier, model_optimizer, model_statedict, optimizer_statedict

# Buid and train model
def build_and_train(model = models.densenet121(pretrained=True), learn_rate = 0.001, input_size = 1024, hidden_size = 256, epochs=10, gpu = True):
    # Define a new, untrained feed-forward
    # network as a classifier, using ReLU activations and dropout

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

        #input_size = 1024
        #hidden_size = hidden_units
        output_size = 102

        #Build Classifier
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                      ('drop', nn.Dropout(0.50)),
                      ('fc1', nn.Linear(input_size, hidden_size)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_size, output_size)),
                      ('softmax', nn.LogSoftmax(dim = 1))
                      ]))

        model.classifier = classifier


    # Train the classifier layers using backpropagation
    # using the pre-trained network to get the features
    if gpu == True:
        model.to('cuda')
    else:
        model.to('cpu')


    #Train a model with a pre-trained network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)


    # Variables
    epochs = epochs
    print_every = 10
    steps = 0
    running_loss = 0

    # Set device agnostic to automatically decide if cuda (gpu) or cpu
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Track the loss and accuracy on the validation set to determine the best hyperparameters
    # Start Training Loop (will measure validation as we're training)
    for e in range(epochs):

        # Make sure training is on
        model.train()

        for ii, (inputs, labels) in enumerate(trainloader):

            steps += 1

            # Move Parameters, model to GPU or CPU, depending on what user requested
            if gpu == True:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            else:
                inputs, labels = inputs.to("cpu"), labels.to("cpu")

            #Flatten image into 25088 (244 * 244) long vector
            #inputs.resize_(inputs.size()[0], 25088)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Training Loss: {:.3F}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3F}.. ".format(valid_loss/len(validloader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()

    return model, model.classifier, optimizer, model.state_dict(), optimizer.state_dict()

#Implement function for validation pass
def validation(model, validloader, criterion, gpu = True):
    valid_loss = 0
    accuracy = 0

    for inputs, labels in validloader:
        if gpu == True:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
        else:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")

        output = model.forward(inputs)
        valid_loss = criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.cuda.FloatTensor).mean()

    return valid_loss, accuracy

def save_checkpoint(checkpoint, save_dir = 'checkpoint.pth'):
    torch.save(checkpoint, save_dir)

# Load the Datasets
data_dir, train_dir, valid_dir, test_dir = load_data(data_directory)

# Transform the Data
train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader = transform_data(data_dir, train_dir, valid_dir, test_dir)

#Set, Build, and Train the Model, Returns info for Checkpoint
model, input_size, model_classifier, model_optimizer, model_statedict, optimizer_statedict = set_model(arch = arch, learn_rate = learn_rate, hidden_units = hidden_units, epochs = epochs, gpu = gpu)

#Create & Save Checkpoint
model.class_to_idx = train_datasets.class_to_idx

checkpoint = {'input_size': input_size,
         'output_size': 102,
         'hidden_size': hidden_units,
         'epochs': epochs,
         'classifier': model_classifier, #model.classifier
         'architecture': arch,
         'state_dict': model_statedict, #model.statedict
         'optimizer': model_optimizer, #optimizer
         'optimizer_state': optimizer_statedict, #optimizer.state_dict()
         'class_to_idx': model.class_to_idx}

save_checkpoint(checkpoint, save_dir)
print("Checkpoint Saved Successfully")
