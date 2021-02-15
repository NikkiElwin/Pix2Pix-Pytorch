from util.dataset import *
from util.models import *
from util.losses import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils

from PIL import Image
import matplotlib.pyplot as plt
import random as ra
import os
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-b",'--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
parser.add_argument("-e",'--epochs', type=int, default=199, metavar='N',
                        help='number of epochs to train (default: 199)')
parser.add_argument("-w",'--workers', type=int, default = 4,
                        metavar='N',help='number of train dataset workers')  
parser.add_argument("-i",'--inverted', type=bool, default = False,
                        metavar='N',help='Whether to flip the training set and test set')                      
parser.add_argument("-lr",'--learning-rate', type=float, default = 2e-4,
                        metavar='N',help='learning rate')        
args = parser.parse_args()


inverted = True  #Source2target or target2source
path = './data/'
BATCH = args.batch_size
EPOCHS = args.epochs
channalsM = 32

trainPath = path + 'train/'
valPath = path + 'val/'
device = "cuda" if torch.cuda.is_available() else "cpu"


netG = UNet(channalsM)
netG.to(device)

netD = Discriminator(channalsM)
netD.to(device)
netD.apply(weights_init)

trainDataset = ProcessDataset(trainPath,SIZE = SIZE,isInverted = inverted)
valDataset = ProcessDataset(valPath,SIZE = SIZE,isInverted = inverted)

trainData = DataLoader(trainDataset,batch_size = BATCH,shuffle = True,num_workers = args.workers)
testData = DataLoader(valDataset,batch_size = BATCH)

optimizerG = optim.Adam(netG.parameters(),lr=args.learning_rate)
optimizerD = optim.Adam(netD.parameters(),lr=args.learning_rate)
#optimizerD = optim.SGD(netD.parameters(), lr = 2e-4, momentum=0.9)

dataiter = iter(testData)
testImgs,testTargets = dataiter.next()
testImgs = testImgs.to(device)
testTargets = testTargets.to(device)
testImgsShow = np.transpose(vutils.make_grid(testImgs.cpu(),nrow=4,padding=2,normalize=True),(1,2,0))
imgs = np.zeros((testImgsShow.shape[0]*2,testImgsShow.shape[1],3))
imgs[:testImgsShow.shape[0],:,:] = testImgsShow

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        
        processBar = tqdm(trainData,ncols = 110,unit="step")
        totalGLoss,totalDLoss = 0.0,0.0
        for step,(sourceImgs,targetImgs) in enumerate(processBar):
            sourceImgs = sourceImgs.to(device)
            targetImgs = targetImgs.to(device)

            netG.zero_grad()
            netD.zero_grad()
            fakeImgs = netG(sourceImgs)
            realPairs = torch.cat([sourceImgs,targetImgs],dim = 1)
            fakePairs = torch.cat([sourceImgs,fakeImgs],dim = 1)

            realOutputs = netD(realPairs)
            fakeOutputs = netD(fakePairs)

            gLoss = lossFunG(fakeOutputs,fakeImgs,targetImgs)
            dLoss = lossFunD(realOutputs,fakeOutputs)
    
            gLoss.backward(retain_graph=True)
            dLoss.backward()
        
            optimizerG.step()
            optimizerD.step()
            
            totalDLoss += dLoss.item()
            totalGLoss += gLoss.item()
            processBar.set_description("[%d/%d] gLoss: %.5f, dLoss: %.5f" % (epoch,EPOCHS,totalGLoss/(step+1),totalDLoss/(step+1)))

        with torch.no_grad():
            plt.figure(figsize=(32,32))
            plt.axis("off")
            fakeImgs = netG(testImgs)
            #fakeImgs = torch.cat([testImgs,fakeImgs],dim = 0)
            fakeImgs = fakeImgs.cpu()
            fakeImgs = np.transpose(vutils.make_grid(fakeImgs,nrow=4,padding=2,normalize=True),(1,2,0))
            
            imgs[testImgsShow.shape[0]:,:,:] = fakeImgs
            plt.imshow(imgs)
            plt.savefig('./Img/Result_epoch % 05d.jpg' % epoch, bbox_inches='tight', pad_inches = 0)
            print('[INFO] Image saved successfully!')

        torch.save(netG.state_dict(), 'model/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'model/netD_epoch_%d.pth' % (epoch))