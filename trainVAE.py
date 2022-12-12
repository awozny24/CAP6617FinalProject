# used to train the VAE model

import os
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from pytorchtool import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tools

import models

slash = '/'

# load source and target data
src = []
trg = []
beginPath = "/home/awozny/SP/DTW/"
filesSource = sorted(glob.glob(beginPath + "src1DTW*"))
filesTarget = sorted(glob.glob(beginPath + "trg1DTW*"))

for i, file in enumerate(filesSource):
    src.append(np.load(file))
    
    if i > 30:
        break;
    
for i, file in enumerate(filesTarget):
    trg.append(np.load(file))
    
    if i > 30:
        break;
    
# put data into numpy array
src = np.concatenate(src)
trg = np.concatenate(trg)

# split data into training and validation set
srcTrain, srcTest, trgTrain, trgTest = train_test_split(src, trg, test_size=0.10, random_state=42)
srcVal, srcTest, trgVal, trgTest = train_test_split(srcTest, trgTest, test_size=0.5, random_state=42)


torch.set_default_tensor_type(torch.DoubleTensor)

# put data into Dataset object
trainDataset = tools.StandardDataset(srcTrain, trgTrain)
valDataset = tools.StandardDataset(srcVal, trgVal)
testDataset = tools.StandardDataset(srcTest, trgTest)

# put data into dataloader
batch_size = 300
trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=False)
valLoader = DataLoader(dataset=valDataset, batch_size=batch_size, shuffle=False)

# get ready for training
train_loss = []
val_loss = []

# set training parameters
Epochs = 50
lr = 0.01
# criteria = models.CustLoss() # ?? change to something else?
criteria = nn.MSELoss()

# define model and send to device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
sizesEnc = [src.shape[1], int(src.shape[1]/2), int(src.shape[1]/3), int(src.shape[1]/4), int(src.shape[1]/8), int(src.shape[1]/16)]
sizesDec = sizesEnc[::-1]
print(sizesEnc)
model = models.VAE(batch_size, sizesEnc, sizesDec).to(device)
print(model)
model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr)

# create directory if it does not exist
numAlbum = 1
album = f"VAE_Try{numAlbum}"
while os.path.exists(os.getcwd() + slash + f"chkpt_{album}"):
    numAlbum += 1
    album = f"VAE_Try{numAlbum}"
print(f"Making Dir: chkpt_{album}")
os.makedirs(os.getcwd() + slash + f"chkpt_{album}")
     
# path to checkpoint model
path = os.getcwd() + slash + "chkpt_"+ album + slash + "checkpoint.pt"

# train model
model, train_loss, val_loss = tools.trainVAEModel(model, optimizer, criteria, trainLoader, valLoader, Epochs=Epochs, path=path, device=device, patience=8)

# load best model
model = models.load_best_model(path, models.VAE(batch_size, sizesEnc, sizesDec))
model = model.to(device)

# calculate validation loss
# do no update gradients
testLossSum = 0
with torch.no_grad():
    model.eval()
    for j, data in enumerate(valLoader):
        # get X and y data
        if device == None:
            testX = data[0]
            testy = data[1]
        else:
            testX = data[0].to(device)
            testy = data[1].to(device)
        
        # calculate validation loss and add to running sum
        model.module.setDecoder('t')
        testLoss = criteria(model(testX), testy)
        testLossSum += testLoss.item()

print(f"Test Loss: {testLossSum/len(valLoader)}")

