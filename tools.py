# https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial/notebook 
import os
import numpy as np
import IPython.display as ipd
from scipy.io.wavfile import write
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
# import librosa
from sklearn.model_selection import train_test_split
from pytorchtool import EarlyStopping
import scipy.io
from simpledtw import dtw

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

def testFunc():
    print("Here")
    

# For help with data loading and preprocessing: 
def GetAudioData(path, fullPath=False):  
    audio = []
    all_audio = []
    sample_rate = []
    all_sample_rate = []
    gender = ['F', 'M']
    
    # if the full path is specified
    if fullPath == True:
        sample_sample_rate, sample_audio = wavfile.read(path)
        sample_rate.append(sample_sample_rate)
        audio.append(sample_audio)
    # if partial path is specified
    else:        
        # for each gender
        for g in gender:
            # for each person of each gender
            for i in range(1, len(gender)+1):
                audio = []
                sample_rate = []
                # get the names of the files containing the audio data
                files = sorted(glob.glob(path + g + str(i) + '/*.wav'))
                            
                # for each file, get the sample and audio data and put in a list
                for f in files:
                    sample_sample_rate, sample_audio = wavfile.read(f)
                    sample_rate.append(sample_sample_rate)
                    audio.append(sample_audio)
                    
                # append set of audios to all audio list
                all_audio.append(audio)
                all_sample_rate.append(sample_rate)
                
     
    # return data
    return all_sample_rate, all_audio


# get mfcc features for each sample
def GetMFCC(audio, sample_rates):
    mfccValsAll = []
    # for each person
    for p, pers in enumerate(audio):
        # for each audio sample from each person
        mfccVals = []
        for a, aud in enumerate(pers):

            mfccVals.append(librosa.feature.mfcc(y=aud.astype(float), sr=22050))

        mfccValsAll.append(mfccVals)
        
    return mfccValsAll


def GetMelSpectogram(audios, sample_rate=22050):
    mfsValsAll = []
    # for each person
    for p, pers in enumerate(audios):
        # for each audio sample from each person
        mfsVals = []
        for a, aud in enumerate(pers):

            mfsVals.append(librosa.feature.melspectrogram(y=aud.astype(float), sr=sample_rate))
            
        mfsValsAll.append(mfsVals)
        
    return mfsValsAll


def MelSpectogramToAudio(melSpectogram):
    return librosa.feature.inverse.mel_to_audio(melSpectogram)


# get fundamental frequency values of audio data
def GetF0(audio):
    F0All = []
    # for each person
    for p, pers in enumerate(audio):
        # for each audio sample from each person
        F0Vals = []
        for a, aud in enumerate(pers):
            # get the fundamental frequency estimaes
            F0Vals.append(pyin(aud.astype(float), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), fill_na=None)[0])

        F0All.append(F0Vals)
        
    return F0All


# get f0 converted from source to target
def GetConvF0(f0Src, mu_src, sigma_src, mu_trg, sigma_trg):
    
    # take the log of the means and variances        
    mu_src_log = log(mu_src)
    sigma_src_log = log(sigma_src)
    mu_trg_log = log(mu_trg)
    sigma_trg_log = log(sigma_trg)
    
    # calculate F0 conversion
    RHS = mu_trg_log + sigma_trg_log/sigma_src_log * (np.log(f0Src.astype(float)) - mu_src_log)
    
    return np.exp(RHS)



def PlayAudioSample(audio, filename='./test', sampleRate = 22050):
    AUDIO_PATH = './' + filename + '.wav'
    
    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(AUDIO_PATH, sampleRate, scaled)

    return ipd.Audio(AUDIO_PATH)



# # perform dynamic time warping on mfcc data
# def ThruDTW(srcData, trgData, addOneHot=False):
#     trgAllDTW = []
#     srcAllDTW = []
    
#     # num people
#     numPeople = len(srcData)
    
#     # for each person
#     for i, src in enumerate(srcData):
#         trgDTW = []
#         srcDTW = []
#         # for each sample phrase from each person
#         for (srcSamp, trgSamp) in zip(src, trgData):
            
#             # get warp path
#             d, wp, steps = librosa.sequence.dtw(trgSamp, srcSamp, metric='euclidean', return_steps=True)
            
#             # get indices for mapping source to target
#             path_x = [p[0] for p in wp[::-1]]
#             path_y = [p[1] for p in wp[::-1]]
            
#             # DTW on source sample
#             newSrc = srcSamp[:, path_y].T
            
#             # one hot encoded vector to append
#             onehotSrc = np.zeros([newSrc.shape[0], len(srcData)])
#             onehotSrc[:, i] = 1
            
#             # store aligned samples to list
#             trgDTW.extend(trgSamp[:, path_x].T)
#             srcDTW.extend(np.concatenate([newSrc, onehotSrc], axis=1))
            
#         # store all samples from each person
#         trgAllDTW.extend(trgDTW)
#         srcAllDTW.extend(srcDTW)
        
#     while len(srcAllDTW) == 1:
#         srcAllDTW = srcAllDTW[0]
        
    return srcAllDTW, trgAllDTW


# perform dynamic time warping on mfcc data
def ThruDTWSingleSource(srcData, trgData):
    trgAllDTW = []
    srcAllDTW = []
    
    count = 0
    
    trgDTW = []
    srcDTW = []

    # for each sample phrase from each person
    for (srcSamp, trgSamp) in zip(srcData, trgData):
        
        # get warp path
        path = dtw(trgSamp, srcSamp)
        path = path[0]
        
        if (count == 0):
            print(path)
            count = 1
        
        # get indices for mapping source to target
        path_trg = [p[0] for p in path]
        path_src = [p[1] for p in path]
        
        # store aligned samples to list
        trgDTW.extend(trgSamp[path_trg, :])
        srcDTW.extend(srcSamp[path_src, :])
        
    # store all samples from each person
    trgAllDTW.append(trgDTW)
    srcAllDTW.append(srcDTW)
        
    while (len(srcAllDTW) == 1) & (len(trgAllDTW) == 1):
        srcAllDTW = srcAllDTW[0]
        trgAllDTW = trgAllDTW[0]
        
    return srcAllDTW, trgAllDTW


# data to put into dataloader
class StandardDataset(Dataset):
    def __init__(self, mfccInput, mfccOutput):
        
        self.X = mfccInput
        self.y = mfccOutput
        
        self.indices = len(self.X)        
        
    def __len__(self):
        return self.indices
      
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    
    
def trainModel(model, optimizer, criteria, trainLoader, valLoader, Epochs=100, path="checkpoint.pt", device=None, patience=10):

    train_loss = []
    val_loss = []

    early_stopping = EarlyStopping(patience=patience, path=path, verbose=True)
    
    f = open("./training_log.txt", 'w')
    f.write("Starting Training:\n")
    f.close()
    
    best_val_loss = 0
    stopCount = 0

    # train model
    for epoch in range(0, Epochs):
        model.train()

        lossSum = 0
        

        for i, data in enumerate(trainLoader):

            # get batch
            if device == None:
                X = data[0]
                y = data[1]
            else:
                X = data[0].to(device)
                y = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # make prediction
            pred = model(X)

            # calculate loss
            loss = criteria(pred, y)

            # compute gradients
            loss.backward()

            # take step
            optimizer.step()

            # add to summed loss
            lossSum += loss.item()

        train_loss.append(lossSum)

        # calculate validation loss
        # do no update gradients
        valLossSum = 0
        with torch.no_grad():
            model.eval()
            for j, vData in enumerate(valLoader):
                # get X and y data
                if device == None:
                    Xval = vData[0]
                    yval = vData[1]
                else:
                    Xval = vData[0].to(device)
                    yval = vData[1].to(device)
                
                # calculate validation loss and add to running sum
                valLoss = criteria(model(Xval), yval)
                valLossSum += valLoss.item()

        # append to running validation loss
        val_loss.append(valLossSum)

        print(f"Epoch {epoch+1}/{Epochs}")
        print(f"\tTrain Loss: {lossSum/len(trainLoader)}")
        print(f"\tVal Loss: {valLossSum/len(valLoader)}")
        
        f = open("./training_log.txt", 'a')
        f.write(f"Epoch {epoch+1}\n\tTrain Loss: {lossSum/len(trainLoader)}\n\tVal Loss: {valLossSum/len(valLoader)}\n")
        f.close()
        
                # use early stopping if needed
        if best_val_loss > valLossSum/len(valLoader):
            best_val_loss = valLossSum/len(valLoader)
            stopCount = 0
            torch.save(model.state_dict(), path)
        else:
            stopCount += 1
            f = open("./training_log.txt", 'a')
            f.write(f"\tEarly Stopping count {stopCount}/{patience}")
            f.close()
            
        if stopCount >= patience:
            f = open("./training_log.txt", 'a')
            f.write(f"Early Stopping. No Improvement. Quitting...")
            f.close()
            return model, train_loss, val_loss

        
    return model, train_loss, val_loss



class KLDiv(nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()
        
        
    def forward(self, z_mean, z_var):
        # loss function is mel cepstral distortion
        return  0.5 * torch.sum(torch.exp(z_var) + torch.square(z_mean) - 1.0 - z_var)  
    
    
def trainVAEModel(model, optimizer, criteria, trainLoader, valLoader, Epochs=100, path="checkpoint.pt", device=None, patience=10):

    train_loss = []
    val_loss = []
    
    KLDiv1  = KLDiv()
    KLDiv2  = KLDiv()
    
    best_val_loss = np.Inf
    stopCount = 0

    early_stopping = EarlyStopping(patience=patience, path=path, verbose=True)
    
    f = open("./training_log.txt", 'w')
    f.write("Starting Training: \n")
    f.close()

    # train model
    for epoch in range(0, Epochs):
        model.train()

        lossSum = 0
        

        for i, data in enumerate(trainLoader):

            # get batch
            if device == None:
                X = data[0]
                y = data[1]
            else:
                X = data[0].to(device)
                y = data[1].to(device)

            # forward and backward pass for source data
            optimizer.zero_grad()
            model.module.setDecoder('s')
            pred, z_mean, z_var = model(X)
            loss1 = criteria(pred, X)
            klloss1 = KLDiv1(z_mean, z_var)
            loss1.backward(retain_graph=True)
            klloss1.backward()
            optimizer.step()
            lossSum += loss1.item() + klloss1
            
            # forward and backward pass for target data
            optimizer.zero_grad()
            model.module.setDecoder('t')
            pred,  z_mean, z_var = model(y)
            loss2 = criteria(pred, y)
            klloss2 = KLDiv2(z_mean, z_var)
            loss2.backward(retain_graph=True)
            klloss2.backward()
            optimizer.step()
            lossSum += loss2.item() + klloss2
            

        train_loss.append(lossSum)

        # calculate validation loss
        # do no update gradients
        valLossSum = 0
        with torch.no_grad():
            model.eval()
            for j, vData in enumerate(valLoader):
                # get X and y data
                if device == None:
                    Xval = vData[0]
                    yval = vData[1]
                else:
                    Xval = vData[0].to(device)
                    yval = vData[1].to(device)
                
                # calculate validation loss and add to running sum
                model.module.setDecoder('s')
                pred, z_mean, z_var = model(Xval)
                valLoss = criteria(pred, Xval)
                valLossSum += valLoss.item()
                model.module.setDecoder('t')
                pred, z_mean, z_var = model(yval)
                valLoss = criteria(pred, yval)
                valLossSum += valLoss.item()

        # append to running validation loss
        val_loss.append(valLossSum)
            
        print(f"Epoch {epoch+1}/{Epochs}")
        print(f"\tTrain Loss: {lossSum/len(trainLoader)}")
        print(f"\tVal Loss: {valLossSum/len(valLoader)}")
        
        f = open("./training_log.txt", 'a')
        f.write(f"Epoch {epoch+1}\n\tTrain Loss: {lossSum/len(trainLoader)}\n\tVal Loss: {valLossSum/len(valLoader)}\n")
        f.close()
        
        # use early stopping if needed
        if best_val_loss > valLossSum/len(valLoader):
            best_val_loss = valLossSum/len(valLoader)
            stopCount = 0
            torch.save(model.state_dict(), path)
        else:
            stopCount += 1
            f = open("./training_log.txt", 'a')
            f.write(f"\tEarly Stopping count {stopCount}/{patience}")
            f.close()
            
        if stopCount >= patience:
            f = open("./training_log.txt", 'a')
            f.write(f"Early Stopping. No Improvement. Quitting...")
            f.close()
            return model, train_loss, val_loss
        
    return model, train_loss, val_loss



def crossCov(a, b):
    mu_a = np.mean(a, axis=0)
    mu_b = np.mean(b, axis=0)
    Sigma_ab = np.zeros([a.shape[1], a.shape[1]])
    for i in range(0, a.shape[0]):
        left = np.asarray([a[i, :].T - mu_a]).T
        right = np.asarray([b[i, :].T - mu_b]).T
        Sigma_ab = left @ right.T
    
    Sigma_ab = Sigma_ab * 1/a.shape[0]
    return Sigma_ab



def Load_mat_Files(path, gender=["F", "M"], persInd=[1, 2]):
    apSource = []
    f0rawSource = []
    n3sgramSource = []
    for g in gender:
        # for each person of each gender
        for i in persInd:
            # features to grab
            ap = []
            f0raw = []
            n3sgram = []

            # get the names of the files containing the audio data
            files = sorted(glob.glob(path + g + str(i) + '*.mat'))
            
            # for each file, get the sample and audio data and put in a list
            for f in files:
                data = scipy.io.loadmat(f)
                ap.append(data["ap"])
                f0raw.append(data["f0raw"])
                n3sgram.append(data["n3sgram"])

            # append set of audios to all audio list
            apSource.append(ap)
            f0rawSource.append(f0raw)
            n3sgramSource.append(n3sgram)

        return apSource, f0rawSource, n3sgramSource


def SP_List_to_np(ap, f0raw, n3sgram):

    # concatenate data into numpy arrays
    apNp = np.concatenate([i.T for i in apSource], axis=0)
    f0rawNp = np.concatenate(f0rawSource, axis=0)
    n3sgramNp = np.concatenate([i.T for i in n3sgramSource], axis=0)

    # concatenate data
    data = np.concatenate([apSrcNp, f0rawSrcNp, n3sgramSrcNp], axis=1)

    return data
