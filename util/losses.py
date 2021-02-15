import torch
import torch.nn as nn


lossF = nn.BCELoss()
LLossF = nn.L1Loss()
def lossFunG(fakeOutputs,fakeImgs,targetImgs,LAMBDA = 100):
    ganLoss = lossF(fakeOutputs,torch.ones_like(fakeOutputs))
    LLoss = LLossF(targetImgs,fakeImgs)
    loss = ganLoss + LAMBDA * LLoss
    return loss

def lossFunD(realOutputs,fakeOutputs):
    realLoss = lossF(realOutputs,torch.ones_like(realOutputs))
    fakeLoss = lossF(fakeOutputs,torch.zeros_like(fakeOutputs))
    
    loss = realLoss + fakeLoss
    return loss