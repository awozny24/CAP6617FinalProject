import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, batchSize, startSize, endSize):
        super(ANN, self).__init__()
        
        self.batchSize = batchSize
        
        # input and output sizes at each layer
        sizes = [startSize, startSize*2, int(startSize*2.5), endSize*2, endSize]
#         sizes = [startSize, , 45, 60, 45, 40, endSize]
#         sizes = [startSize, 64, 32, 32, 64, endSize]
#         sizes = [startSize, 256, 256, endSize]
        
        # define layers
        self.layers = nn.ModuleList()
        for i in range(0, len(sizes)-2):
            self.layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i+1]),
                                    nn.Tanh())
                              )
        self.layers.append(nn.Sequential(nn.Linear(sizes[-2], sizes[-1])))
                
    
    def forward(self, x):
        
        out = x
        for i in range(0, len(self.layers)):
            torch.nn.functional.normalize(out)
            out = self.layers[i](out)
         
        return out
        
        
        
def ToMelSpectralDomain(f):
    mel = 1127.01048 * torch.log(f/700 + 1) 
    return mel
        
        
        
class VAE(nn.Module):
    def __init__(self, batchSize, sizesEnc, sizesDec):
        super(VAE, self).__init__()
        
        self.batchSize = batchSize
        
        # make encoder and decoder layers
        self.Enc = self.MakeLayers(sizesEnc)
        self.Dec1 = self.MakeLayers(sizesDec)
        self.Dec2 = self.MakeLayers(sizesDec)

                
    def MakeLayers(self, sizes):
        
        layers = nn.ModuleList()
        for i, val in enumerate(sizes):
            if i == len(sizes) - 1:
                break
            
            if i != len(sizes)-2:
                layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i+1], bias=True),
                                    nn.ReLU()))
            else:
                layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i+1], bias=True)))
                
        return layers
        
    
    def setDecoder(self, decoder):
        self.decoder = decoder
                
    
    def forward(self, x):
        
        out = x
        
        # pass first dataset through first encoder
        for layer in self.Enc:
            torch.nn.functional.normalize(out)
            out = layer(out)
            
        z = out

        if 's' in self.decoder:
            # pass first dataset through first decoder
            for layer in self.Dec1:
                out = layer(out)    
        else:
            # pass first dataset through second decoder
            for layer in self.Dec2:
                out = layer(out)
       
        return out, torch.mean(z, axis=0), torch.var(z, axis=0)
        


# this class solution is from: https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379
# wrapper to load model
# nn.DataParallel() was used during training. If using w/ different configuration
# use this wrapper to load the model
class WrappedModel(nn.Module):
	def __init__(self, model):
		super(WrappedModel, self).__init__()
		self.module = model
	def forward(self, x):
		return self.module(x)


          
class CustLoss(nn.Module):
    def __init__(self):
        super(CustLoss, self).__init__()
        
        
    def forward(self, output, label):
        # loss function is mel cepstral distortion
        return  torch.sum(torch.square(torch.sum(ToMelSpectralDomain(output) - ToMelSpectralDomain(label), axis=1)))
        


# function to load model
def load_best_model(path, modelArch):
    try:
        model = modelArch
        model.load_state_dict(torch.load(path))
    except: 
        try:
            model = modelArch
            model.load_state_dict(torch.load(path), map_location=torch.device('cpu'))
        except:
            try: 
                model = WrappedModel(modelArch)
                model.load_state_dict(torch.load(path))
            except:
                model = WrappedModel(modelArch)
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    return model

      