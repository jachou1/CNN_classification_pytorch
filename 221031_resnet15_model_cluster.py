import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset

from glob import glob
imagePatches = glob('/Users/jacquelinechou/ctap_small/**/*.jpg', recursive=True)

# Get the labels
import fnmatch
patternZero = '*EFM.jpg'
patternOne = '*TF.jpg'
classZero = fnmatch.filter(imagePatches, patternZero)
classOne = fnmatch.filter(imagePatches, patternOne)

y = []
for img in imagePatches:
    if img in classZero:
        y.append(0)
    elif img in classOne:
        y.append(1)
#     if len(y) > 35000:
#         break

images_df = pd.DataFrame()
images_df["images"] = imagePatches #[:35001]
images_df["labels"] = y

# get the test set first (20% of data)
main, test= train_test_split(images_df, stratify = images_df.labels, test_size = 0.2)

# get the train and validation sets (60% train, 20% test)
train, val = train_test_split(main, stratify=main.labels, test_size=0.2)

from torchvision.io import read_image

class MyDataset(Dataset):
    def __init__(self, df_data, transform=None):
        super().__init__()
        self.df = df_data.values
        
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path,label = self.df[index]
        
        image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        image = torchvision.transforms.Resize((30,30))(image) # should probably be (30,30) no resizing
        if self.transform is not None:
             image = self.transform(image)
        return image, label


# Hyper parameters
n_epochs = 100
num_classes = 2
batch_size = 128
# learning_rate = 0.002

# Use transforms.compose method to reformat images for modeling,
trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20), 
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
# Create Training and Test dataset
train_dataset = MyDataset(df_data=train, transform=trans_train)
valid_dataset = MyDataset(df_data=val,transform=trans_valid)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size =batch_size,
                                          shuffle= True)
test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size,
                                         shuffle= False)

def set_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    return torch.device(dev)

def train_nn(model, train_loader, test_loader, optimizer, n_epochs, lr):
    device = set_device()
    best_f1_score = 0

    train_losses = []   # holds the train_losses per epoch, list of 100 numbers
    train_accuracy = [] # holds the train_accuracy per epoch, list of 100 numbers
    train_f1 = []       # holds the f1 scores per epoch for class_1

    validation_losses = []
    validation_accuracy = []
    validation_f1 = []
    
    for epoch in range(n_epochs):
        print('Epoch number %d' % (epoch + 1))
        model.train() # train the model
        running_loss = 0.0
        running_correct = 0.0
        running_class0_correct = 0.0
        running_class1_correct = 0.0
        running_class0_FN = 0.0
        running_class1_FP = 0.0
        total = 0
        
        for data in train_loader: # goes through the training batches
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            # set gradient to zero initially and for each batch
            optimizer.zero_grad()
            
            outputs = model(images).squeeze()
            
            #_, predicted = torch.max(outputs.data, 1) # outputs.data = tensor behind the object
            
            predicted = (outputs > 0).int()
#             print(predicted.dtype)
#             print(labels.dtype)
            # hard-coded that the criterion is focal loss
            loss = torchvision.ops.sigmoid_focal_loss(outputs, labels.float(), gamma = 5, reduction = 'sum')
            
            loss.backward() # calculates the gradients
            #print(loss)
            optimizer.step() # backpropagate to get weights
            
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
            # count the number of correctly classified class_0
            running_class0_correct += ((labels == predicted) & (predicted == 0)).sum().item()
            # count the number of correctly classified class_1
            running_class1_correct += ((labels == predicted) & (predicted == 1)).sum().item()
            # count the number of incorrectly classified class_1 (FP)
            running_class1_FP += ((predicted == 1) & (labels != predicted)).sum().item()
            # count the number of incorrectly classified class_0 (FN)
            running_class0_FN += ((predicted == 0) & (labels != predicted)).sum().item()
        
        epoch_loss = running_loss/len(train_loader)       # calculates the epoch loss
        train_losses.append(epoch_loss)
        epoch_accuracy = 100.00 * running_correct/total   # calculates the epoch accuracy
        train_accuracy.append(epoch_accuracy)
        # F1_score_class_1 = TP/ (TP + 0.5(FP + FN)), where FP = false positive
        epoch_f1_score_class_1 = running_class1_correct/(running_class1_correct + 0.5*(running_class0_FN + running_class1_FP))
        train_f1.append(epoch_f1_score_class_1)

        ## WILL I BE PRINTING ANYTHING?
        print(' -Training dataset. Got %d out of %d images correctly (%3f%%). Epoch loss: %.3f'
             % (running_correct, total, epoch_accuracy, epoch_loss))
        
        # better practice to write an individual function for each thing you want to do
        test_dataset_accuracy, test_dataset_f1, test_dataset_loss = evaluate_model_on_test_set(model, test_loader)
        validation_accuracy.append(test_dataset_accuracy)
        validation_f1.append(test_dataset_f1)
        validation_losses.append(test_dataset_loss)

        # Want to save the model with best F1 Score for class_1 
        if (test_dataset_f1 > best_f1_score):
            best_f1_score = test_dataset_f1
            save_checkpoint(model, epoch, optimizer, best_f1_score, lr)
            
    # save the lists of values
    np.save('/Users/jacquelinechou/train_losses', train_losses)    
    np.save('/Users/jacquelinechou/train_accuracy', train_accuracy) 
    np.save('/Users/jacquelinechou/train_f1', train_f1) 
    np.save('/Users/jacquelinechou/validation_losses', validation_losses) 
    np.save('/Users/jacquelinechou/validation_accuracy', validation_accuracy) 
    np.save('/Users/jacquelinechou/validation_f1', validation_f1) 
    #print('Finished')
    #return model

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    running_loss = 0
    predicted_correctly_on_epoch = 0
    predicted_correctly_class_0 = 0
    predicted_correctly_class_1 = 0
    predicted_class_0_wrong = 0       # calling this false negative (FN)
    predicted_class_1_wrong = 0       # calling this false positive (FP)
    total = 0
    device = set_device()
    
    with torch.no_grad():
        for data in test_loader:          # batches of validation dataset
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)       # keeps tabs on how many images have been passed through
            
            outputs = model(images).squeeze()
            
            predicted = (outputs > 0).int()
            
            #_, predicted = torch.max(outputs.data,1)
            
            loss = torchvision.ops.sigmoid_focal_loss(outputs, labels.float(), gamma = 5, reduction = 'sum')
            running_loss += loss.item()
            
            # calculate correct predictions
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
            predicted_correctly_class_0 += ((predicted == labels) & (labels == 0)).sum().item()
            predicted_correctly_class_1 += ((predicted == labels) & (labels == 1)).sum().item()
            
            # calculate false_negatives, false_positives
            predicted_class_0_wrong += ((predicted != labels) & (labels == 0)).sum().item()
            predicted_class_1_wrong += ((predicted != labels) & (labels == 1)).sum().item()
        
    epoch_accuracy = 100.00 * predicted_correctly_on_epoch/total
    print(' -Testing dataset. Got %d out of %d images correctly (%.3f%%)'
         % (predicted_correctly_on_epoch, total, epoch_accuracy))

    ### write equation for epoch_test_f1
    # F1_score_class_1 = TP/ (TP + 0.5(FP + FN))
    epoch_test_f1 = predicted_correctly_class_1/(predicted_correctly_class_1 + 0.5*(predicted_class_0_wrong + predicted_class_1_wrong))
    
    epoch_test_loss = running_loss/len(test_loader)       # calculates the epoch loss 
    
    return epoch_accuracy, epoch_test_f1, epoch_test_loss

def save_checkpoint(model, epoch, optimizer, best_accuracy, lr):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best_f1': best_f1,
        'optimizer': optimizer.state_dict(),
        'comments': 'very cool model'
    }
    torch.save(state, f'TF_EFM_test_model_gamma_5_lr_{lr}_epoch.pth.tar')

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

resnet18_model = models.resnet18(pretrained = True)
num_features = resnet18_model.fc.in_features
number_of_classes = num_classes

resnet18_model.fc = nn.Linear(num_features, 1) # num_of_classes instead of 1 for multi-class classification
device = set_device()
resnet18_model = resnet18_model.to(device)

# try several values of the learning_rate increase in performance
learning_rate = np.array([0.2, 0.02, 0.002, 0.0002, 0.000002])

# loss_fn = torchvision.ops.sigmoid_focal_loss(gamma = 5)
optimizer = optim.SGD(resnet18_model.parameters(), lr = learning_rate[0], momentum= 0.9,
                     weight_decay = 0.003)

train_nn(resnet18_model, train_loader, test_loader, optimizer, n_epochs = n_epochs, lr = learning_rate[0])

