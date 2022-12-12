# takes feature matrices representing speaker and target audio files and calls function
# to perform DTW on them so that they line up

import numpy as np
from simpledtw import dtw
import pickle
from tools import ThruDTWSingleSource

trg = []
trg = np.load("/home/awozny/SP/trg1.npy")

src = []
src = np.load("/home/awozny/SP/src1.npy")


with open('/home/awozny/SP/srcIndices.pkl', 'rb') as pickle_file:
    srcIndices = pickle.load(pickle_file)
    
with open('/home/awozny/SP/trgIndices.pkl', 'rb') as pickle_file:
    trgIndices = pickle.load(pickle_file)
  
# separate each speech sample into a list for target  
ind = 0
srcSep = []
for i in srcIndices:
    srcSep.append(src[ind:ind+i, :])
    ind += i
   
# separate each speech sample into a list for target
ind=  0 
trgSep = []
for i in trgIndices:
    trgSep.append(src[ind:ind+i, :])
    ind += i
    
print(len(srcSep))
print(len(trgSep))
    
newSrc, newTrg = ThruDTWSingleSource(srcSep, trgSep)

np.save('/home/awozny/SP/src1DTW.npy', newSrc)
np.save('/home/awozny/SP/trg1DTW.npy', newTrg)

print(newSrc.shape)
print(newTrg.shape)
