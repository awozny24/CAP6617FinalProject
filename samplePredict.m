% loads .mat files for predicted STRAIGHT features and synthesizes speech

[x, fs] = audioread('../../voice-conversion/vcc2018_training/VCC2TF1/10001.wav'); 
f0raw1 = MulticueF0v14(x,fs); 
ap1 = exstraightAPind(x,fs,f0raw1);
n3sgram1=exstraightspec(x,f0raw1,fs);
sy1 = exstraightsynth(f0raw1, n3sgram1, ap1, fs)

load('samplePredANN.mat')
load('samplePredVAE.mat')

fs = 22050;
sy = exstraightsynth(f0raw1, n3sgram1, ap1, fs);
sound(sy, fs);