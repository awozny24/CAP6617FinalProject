## CAP6617FinalProject
#Voice Conversion
Aims to convert a sentence uttered by a source speaker to sound like the voice of the target speaker saying it.

GetFeatures.m is a Matlab file that uses the STRAIGHT vocoder library to extract audio features from the source and target speakers.

mat_to_npy.py loads the .mat files produced from GetFeatures.m, converts them into one numpy matrix and saves them

simpledtw.py is a file used to perform dynamic time warping on the source and target speaker data. Taken from https://github.com/talcs/simpledtw

DTW.py is a file that calls the functions necessary to perform dynamic time warping on the data to save it

tools.py contains tools used to work with the data and models. Includes functions to train models.

models.py contains the model class defitions and function for loading the best models

Test.py loads the trained models and makes predictions on a loaded audio sample.

trainANN.py and trainVAE.py are files used to setup training for and training the ANN and VAE models.

samplePredict.m takes the .mat file generated from the predictions in Test.py and uses the STRAIGHT vocoder library to synthesize speech from the predicted features

 Note that the .mat and .npy created from running some of these files are not located on here since they take up a lot of space.  
 The data used for this project can be found at https://datashare.ed.ac.uk/handle/10283/3061


Functions to extract STRAIGHT Features were taken from https://github.com/HidekiKawahara/legacy_STRAIGHT.  These libraries are written in Matlab.
