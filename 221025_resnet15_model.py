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

# Pull in images from within the server
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

images_df = pd.DataFrame()
images_df["images"] = imagePatches
images_df["labels"] = y

# get the test set first (20% of data)
main, test= train_test_split(images_df, stratify = images_df.labels, test_size
													test_size = 0.2)
# get the train and validation sets (60% train, 20% test)
train, val = train_test_split(main, stratify=images_df.labels, test_size=0.2)

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

## Parameters for model

# Hyper parameters
n_epochs = 100
num_classes = 2
batch_size = 128
# learning_rate = 0.002

# try several values of the learning_rate increase in performance
learning_rate = np.array([0.2, 0.02, 0.002, 0.0002, 0.000002])

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


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()


def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_accuracy = 0

    train_losses = [] 	# holds the train_losses per epoch, list of 100 numbers
    train_accuracy = [] # holds the train_accuracy per epoch, list of 100 numbers
    train_f1 = [] 		# holds the f1 scores per epoch for class_1

    validation_losses = []
    validation_accuracy = []
    validation_f1 = []
    
    for epoch in range(n_epochs):
        print('Epoch number %d' % (epoch + 1))
        model.train() # train the model
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        
        for data in train_loader: # goes through the training batches
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            #set gradient to zero initially
            optimizer.zero_grad()
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels, gamma)
            
            loss.backward() # backpropagate to get weights
            
            optimizer.step()
            
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
        
        epoch_loss = running_loss/len(train_loader) 		# calculates the epoch loss
        train_losses.append(epoch_loss)
        epoch_accuracy = 100.00 * running_correct/total		# calculates the epoch accuracy
        train_accuracy.append(epoch_accuracy)
        # F1_score_class_1 = TP/ (TP + 0.5(FP + FN)), where FP = false positive
        epoch_f1_score_class_1 = running_class1_correct/(running_class1_correct + 0.5(running_class0_FN + running_class1_FP))


       	## WILL I BE PRINTING ANYTHING?
        print(' -Training dataset. Got %d out of %d images correctly (%3f%%). Epoch loss: %.3f'
             % (running_correct, total, epoch_accuracy, epoch_loss))
        
        test_dataset_accuracy, test_dataset_f1, test_dataset_loss = evaluate_model_on_test_set(model, test_loader)
        validation_accuracy.append(test_dataset_accuracy)
        validation_f1.append(test_dataset_f1)
        validation_f1.append(test_dataset_loss)

        # if (test_dataset_accuracy > best_accuracy):
        #     best_accuracy = test_dataset_accuracy
        # Want to save the model at each epoch
        save_checkpoint(model, epoch, optimizer, best_accuracy)
        
    print('Finished')
    return model

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data,1)
            
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
        
    epoch_accuracy = 100.00 * predicted_correctly_on_epoch/total
    print(' -Testing dataset. Got %d out of %d images correctly (%.3f%%)'
         % (predicted_correctly_on_epoch, total, epoch_accuracy))

    ### write equation for epoch_test_f1
    
    return epoch_accuracy, epoch_test_f1, test_dataset_loss



def save_checkpoint(model, epoch, optimizer, best_accuracy):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best_accuracy': best_accuracy,
        'optimizer': optimizer.state_dict(),
        'comments': 'very cool model'
    }
    torch.save(state, f'TF_EFM_test_model_gamma_5_{lr}_epoch_{epoch+1}.pth.tar')

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

resnet18_model = models.resnet18(pretrained = True)
num_features = resnet18_model.fc.in_features
number_of_classes = num_classes

resnet18_model.fc = nn.Linear(num_features, number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device)

# use a weighted cross-entropy loss to account for imbalanced classes -> this was hard-coded, but these weights are 
# not necessarily true in each batch during training
# loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([1/18699, 1/5301]))

# let's fix gamma = 5 for now
loss_fn = torchvision.ops.sigmoid_focal_loss(gamma = 5)

optimizer = optim.SGD(resnet18_model.parameters(), lr = learning_rate, momentum= 0.9,
                     weight_decay = 0.003)

train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, n_epochs = n_epochs)






















