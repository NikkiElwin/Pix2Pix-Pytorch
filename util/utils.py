import torch
import torch.nn as nn


class DownSample(nn.Module):
    def __init__(self,inChannals,outChannals):
        super(DownSample,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inChannals,out_channels=outChannals,kernel_size=3,stride=1,
                      padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outChannals),
            nn.LeakyReLU(0.2,True),
            
            nn.Conv2d(in_channels=outChannals,out_channels=outChannals,kernel_size=3,stride=1,
                      padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outChannals),
            nn.LeakyReLU(0.2,True),
            
            nn.Conv2d(in_channels=outChannals,out_channels=outChannals,kernel_size=3,stride=1,
                      padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outChannals),
            nn.LeakyReLU(0.2,True)
        )
        self.maxPool = nn.MaxPool2d(2)
        
        
    def forward(self,input):
        residual = self.model(input)
        out = self.maxPool(residual)
        return out,residual
    
    
class UpSample(nn.Module):
    def __init__(self,inChannals,outChannals):
        super(UpSample,self).__init__()   
        self.upSample = nn.ConvTranspose2d(in_channels=inChannals,out_channels=inChannals,
                               kernel_size=4,stride=2,padding=1)
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inChannals * 2,out_channels=outChannals,kernel_size=3,stride=1,
                      padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outChannals),
            nn.LeakyReLU(0.2,True),
            
            nn.Conv2d(in_channels=outChannals,out_channels=outChannals,kernel_size=3,stride=1,
                      padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outChannals),
            nn.LeakyReLU(0.2,True),
            
            nn.Conv2d(in_channels=outChannals,out_channels=outChannals,kernel_size=3,stride=1,
                      padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outChannals),
            nn.LeakyReLU(0.2,True)
        )
        
    def forward(self,input,residual):
        x = self.upSample(input)
        x = torch.cat([x,residual],dim = 1)
        x = self.model(x)
        return x