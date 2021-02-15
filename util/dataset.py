import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from PIL import Image
import numpy as np
import random as ra 

SIZE = 256

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    transforms.Resize([SIZE,2*SIZE])
])

class ProcessDataset(Dataset):
    
    def __init__(self,path,transform = transform,SIZE = SIZE,isInverted = False):
        self.allImgPaths = []
        for root,_,files in os.walk(path):
            self.allImgPaths = [root + file for file in files]
            
        self.isInverted = isInverted
        self.SIZE = SIZE
        self.path = path
        self.transform = transform
        self.length = len(self.allImgPaths)
        ra.shuffle(self.allImgPaths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        _img = self.allImgPaths[index]
        _img = Image.open(_img)
        
        _img = self.transform(_img)
        
        sourceImg = _img[:,:,:self.SIZE] if not self.isInverted else _img[:,:,self.SIZE:]
        targetImg = _img[:,:,self.SIZE:] if not self.isInverted else _img[:,:,:self.SIZE]
        return sourceImg,targetImg