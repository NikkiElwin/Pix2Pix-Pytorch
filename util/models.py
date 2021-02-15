import torch
import torch.nn as nn

from .utils import *

class UNet(nn.Module):
    def __init__(self,channalsM = 64,zoom = 4):
        super(UNet,self).__init__()
        
        _downSample = [DownSample(3,channalsM)]
        _upSample = [UpSample(channalsM,3)]
        for i in range(zoom-1):
            _downSample.append(DownSample(channalsM * 2**i,channalsM * 2**(i+1)))
            _upSample.insert(0,UpSample(channalsM * 2**(i+1),channalsM * 2**i))
            
        """     
        self.downSample = nn.ModuleList([
            DownSample(3,64),
            DownSample(64,128),
            DownSample(128,256),
            DownSample(256,512),
        ])
        self.upSample = nn.ModuleList([
            UpSample(512,256),
            UpSample(256,128),
            UpSample(128,64),
            UpSample(64,3)
        ])
        """
        
        self.downSample = nn.ModuleList(_downSample)
        self.upSample = nn.ModuleList(_upSample)
        self.output = nn.Sigmoid()
            
    def forward(self,x):
        residuals = []
        for block in self.downSample:
            x,_residual = block(x)
            residuals += [_residual]
        
        for block in self.upSample:
            x = block(x,residuals.pop(-1))
        
        #x = self.output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,channalsM = 64):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 6,out_channels = channalsM,kernel_size = 1,stride = 1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(in_channels = channalsM,out_channels = channalsM * 2,kernel_size = 1,stride = 1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(in_channels = channalsM * 2,out_channels = 1,kernel_size = 1,stride = 1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.model(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data,mean=0.0,std=0.02)