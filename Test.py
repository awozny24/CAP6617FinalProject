# This file takes features from an audio sample file (saved in .mat)
# and makes predictions on it using the trained ANN and VAE models

import numpy as np
import torch
import torch.nn as nn
import scipy.io
import models
import importlib

importlib.reload(models)

import matplotlib.pyplot as plt


#### ANN ####
sample = scipy.io.loadmat("./SP/Source/F1_01.mat")
ap = sample["ap"]
f0raw = sample["f0raw"]
n3sgram = sample["n3sgram"]
X = np.concatenate([ap.T, f0raw, n3sgram.T], axis=1)

model = models.ANN(X.shape[1], X.shape[1])
model = models.load_best_model("ANN_checkpoint.pt", model)

with torch.no_grad():
    model.eval()
    pred = model(torch.Tensor(X))
    
pred = pred.numpy()
apPredANN = pred[:, :1025]
f0rawPredANN = pred[:, 1025]
f0rawPredANN = np.asarray([f0rawPred])
n3sgramPredANN = pred[:, 1026:]

# scipy.io.savemat("samplePredANN.mat", {"ap":apPred.T, "f0raw":f0rawPred.T, "n3sgram":n3sgramPred.T})


#### VAE #### 
sample = scipy.io.loadmat("./SP/Source/F1_01.mat")
ap = sample["ap"]
f0raw = sample["f0raw"]
n3sgram = sample["n3sgram"]
X = np.concatenate([ap.T, f0raw, n3sgram.T], axis=1)

st = 2051
sizesEnc = [st, int(st/2), int(st/4), int(st/8)]
sizesDec = sizesEnc[::-1]
model = models.VAE_Simple(sizesEnc, sizesDec)
model = models.load_best_model("VAE_checkpoint.pt", model)

with torch.no_grad():
    model.eval()
    model.module.setDecoder('t')
    pred = model(torch.Tensor(X))
    
pred = pred.numpy()
apPredVAE = pred[:, :1025]
f0rawPredVAE = pred[:, 1025]
f0rawPredVAE = np.asarray([f0rawPred])
n3sgramPredVAE = pred[:, 1026:]

# scipy.io.savemat("samplePredVAE.mat", {"ap":apPred.T, "f0raw":f0rawPred.T, "n3sgram":n3sgramPred.T})


# get aperiodicity plots
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(25)

plt.subplot(3, 1, 1)
plt.imshow(ap)

plt.subplot(3, 1, 2)
plt.imshow(apPredANN.T)

plt.subplot(3, 1, 3)
plt.imshow(apPredVAE.T)
plt.show()


# get fundamental frequency plots
fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(15)

plt.subplot(1, 3, 1)
plt.plot(range(0, len(f0raw.squeeze())), f0raw.squeeze())

plt.subplot(1, 3, 2)
plt.plot(range(0, len(f0rawPredANN.squeeze())), f0rawPredANN.squeeze())

plt.subplot(1, 3, 3)
plt.plot(range(0, len(f0rawPredVAE.squeeze())), f0rawPredVAE.squeeze())

plt.show()


# get STRAIGHT spectogram plots
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(25)

plt.subplot(3, 1, 1)
plt.imshow(n3sgram)

plt.subplot(3, 1, 2)
plt.imshow(n3sgramPredANN.T)

plt.subplot(3, 1, 3)
plt.imshow(n3sgramPredVAE.T)
plt.show()