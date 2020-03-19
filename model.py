import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.transforms import *



class LStageBlock(nn.Module):
    def __init__(self,firstlayerinputdim=128, blockinputdim=96):
        super(LStageBlock, self).__init__()
        self.c2d1 = nn.Conv2d(firstlayerinputdim, blockinputdim, 3, padding=1)
        self.prl1 = nn.PReLU()
        self.c2d2 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        self.prl2 = nn.PReLU()
        self.c2d3 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        self.prl3 = nn.PReLU()
    def forward(self, input):
        o1 = self.prl1(self.c2d1(input))
        o2 = self.prl2(self.c2d2(o1))
        o3 = self.prl3(self.c2d3(o2))
        o = torch.cat((o1,o2,o3),1)
        return o

class LStage(nn.Module):
    def __init__(self, stageinputdim=128, blockinputdim=96, penultimatelayerdim=256):
        super(LStage, self).__init__()
        self.b1 = LStageBlock(stageinputdim, blockinputdim)
        self.b2 = LStageBlock(blockinputdim*3, blockinputdim)
        self.b3 = LStageBlock(blockinputdim*3, blockinputdim)
        self.b4 = LStageBlock(blockinputdim*3, blockinputdim)
        self.b5 = LStageBlock(blockinputdim*3, blockinputdim)
        self.penultimatelayer = nn.Conv2d(blockinputdim*3, penultimatelayerdim, 1, padding=0)
        self.prlpl = nn.PReLU()
        self.ouputlayer = nn.Conv2d(penultimatelayerdim, 52, 1, padding=0)

    def forward(self, input):
        o = self.b1(input)
        o = self.b2(o)
        o = self.b3(o)
        o = self.b4(o)
        o = self.b5(o)
        o = self.prlpl(self.penultimatelayer(o))
        o = self.ouputlayer(o)
        return o

class SStageBlock(nn.Module):
    def __init__(self,firstlayerinputdim=128, blockinputdim=96):
        super(SStageBlock, self).__init__()
        self.c2d1 = nn.Conv2d(firstlayerinputdim, blockinputdim, 3, padding=1)
        self.prl1 = nn.PReLU()
        self.c2d2 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        self.prl2 = nn.PReLU()
        self.c2d3 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        self.prl3 = nn.PReLU()
    def forward(self, input):
        o1 = self.prl1(self.c2d1(input))
        o2 = self.prl2(self.c2d2(input))
        o3 = self.prl3(self.c2d3(input))
        o = torch.cat((o1,o2,o3),1)
        return o

class SStage(nn.Module):
    def __init__(self, stageinputdim=128, blockinputdim=96, penultimatelayerdim=256):
        super(SStage, self).__init__()
        self.b1 = SStageBlock(stageinputdim, blockinputdim)
        self.b2 = SStageBlock(blockinputdim, blockinputdim)
        self.b3 = SStageBlock(blockinputdim, blockinputdim)
        self.b4 = SStageBlock(blockinputdim, blockinputdim)
        self.b5 = SStageBlock(blockinputdim, blockinputdim)
        self.penultimatelayer = nn.Conv2d(blockinputdim*3, penultimatelayerdim, 1, padding=0)
        self.prlpl = nn.PReLU()
        self.ouputlayer = nn.Conv2d(penultimatelayerdim, 52, 1, padding=0)

    def forward(self, input):
        o = self.b1(input)
        o = self.b2(o)
        o = self.b3(o)
        o = self.b4(o)
        o = self.b5(o)
        o = self.prlp(self.penultimatelayer(o))
        o = self.ouputlayer(o)
        return o





class OpenPoseModel(nn.Module):

    def __init__(self):
        super(OpenPoseModel, self).__init__()
        self.F = self.__init_features_layerset()
        #LStages
        self.L = LStage(180, 128, 512) #128, 96 ,256

        #SStages
        self.S = SStage(206, 128, 512)#180, 96, 256


    #this is the schema reflected in prototxt of the cmu-perceptual implementation
    def __init_features_layerset(self):
        od = OrderedDict()
        od['0'] = nn.Conv2d(3, 64, 3, padding=1)
        od['1'] = nn.ReLU()
        od['2'] = nn.Conv2d(64,64,3,padding=1)
        od['3'] = nn.ReLU()
        od['4'] = nn.MaxPool2d(2,stride=2)

        od['5'] = nn.Conv2d(64,128,3, padding=1)
        od['6'] = nn.ReLU()
        od['7'] = nn.Conv2d(128,128, 3, padding=1)
        od['8'] = nn.ReLU()
        od['9'] = nn.MaxPool2d(2, stride=2)

        od['10'] = nn.Conv2d(128, 256, 3, padding=1)
        od['11'] = nn.ReLU(inplace=True)
        od['12'] = nn.Conv2d(256, 256, 3, padding=1)
        od['13'] = nn.ReLU(inplace=True)
        od['14'] = nn.Conv2d(256, 256, 3, padding=1)
        od['15'] = nn.ReLU(inplace=True)
        od['16'] = nn.Conv2d(256, 256, 3, padding=1)
        od['17'] = nn.ReLU(inplace=True)
        od['18'] = nn.MaxPool2d(2, stride=2)

        od['19'] = nn.Conv2d(256, 512, 3, padding=1)
        od['20'] = nn.ReLU(inplace=True)
        od['21'] = nn.Conv2d(512, 512, 3, padding=1)
        od['22'] = nn.ReLU(inplace=True)
        od['23'] = nn.Conv2d(512,256, 3, padding=1)
        od['24'] = nn.PReLU()
        od['25'] = nn.Conv2d(256, 128, 3, padding=1)
        od['26'] = nn.PReLU()

        return nn.Sequential(od)

    def forward_L(self, features, passindex=0):
        #start L stages : 4 stages
        input = None
        if (passindex==0):
            #attach 52 filters of the given dimension set to zero
            att = torch.zeros((features.shape[0], 180 - features.shape[1], features.shape[2], features.shape[3]))
            input = torch.cat((features, att), 1)
        else:
            input = features
        Ls = self.L(input)
        flattened = torch.flatten(Ls)
        return Ls, flattened

    def forward_S(self, L4, features):
        Ss = torch.cat((L4, features), 1)
        Ss = self.S1(Ss)
        Ss = torch.cat((Ss, L4, features), 1)
        Ss = self.S2(Ss)
        flattened = torch.flatten(Ss)
        return Ss, flattened

    def Fparameters(self):
        return self.F.parameters()

    def Lparameters(self):
        parameters = list(self.L1.parameters()) + list(self.L2.parameters()) + \
                     list(self.L3.parameters()) + list(self.L4.parameters()) + list(self.Ldecoder.parameters())
        return parameters

    def Sparameters(self):
        parameters = list(self.S1.parameters()) + list(self.S2.parameters()) + list(self.Sdecoder.parameters())
        return parameters

    # def __init_Lstage(self, stageinputdim=128, blockinputdim=96, penultimatelayerdim=256):#inputdim is 128 at first go, and 288 at the rest (96*3)
    #
    #     lod = OrderedDict()
    #
    #     #1
    #     od = OrderedDict()
    #     od['01'] = nn.Conv2d(stageinputdim,blockinputdim,3,padding=1)
    #     od['011'] = nn.PReLU()
    #     od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['021'] = nn.PReLU()
    #     od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['031'] = nn.PReLU()
    #
    #     lod['0'] = nn.Sequential(od)
    #
    #     #2
    #     od = OrderedDict()
    #     od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['011'] = nn.PReLU()
    #     od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['021'] = nn.PReLU()
    #     od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['031'] = nn.PReLU()
    #
    #     lod['1'] = nn.Sequential(od)
    #
    #     # 3
    #     od = OrderedDict()
    #     od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['011'] = nn.PReLU()
    #     od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['021'] = nn.PReLU()
    #     od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['031'] = nn.PReLU()
    #
    #     lod['2'] = nn.Sequential(od)
    #
    #     # 4
    #     od = OrderedDict()
    #     od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['011'] = nn.PReLU()
    #     od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['021'] = nn.PReLU()
    #     od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['031'] = nn.PReLU()
    #
    #     lod['3'] = nn.Sequential(od)
    #
    #     # 5
    #     od = OrderedDict()
    #     od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['011'] = nn.PReLU()
    #     od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['021'] = nn.PReLU()
    #     od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['031'] = nn.PReLU()
    #
    #     lod['4'] = nn.Sequential(od)
    #
    #     lod['5'] = nn.Conv2d(blockinputdim*3, penultimatelayerdim, 1, padding=0)
    #     lod['6'] = nn.PReLU()
    #     lod['7'] = nn.Conv2d(penultimatelayerdim, 52, 1, padding=0)
    #
    #     return nn.Sequential(lod)
    #
    #
    # def __init_Sstage(self, stageinputdim=128, blockinputdim=96, penultimatelayerdim=256):  # inputdim is 128 at first go, and 288 at the rest (96*3)
    #     sod = OrderedDict()
    #     #1
    #     od = OrderedDict()
    #     od['0'] = nn.Conv2d(stageinputdim, blockinputdim, 3, padding=1)
    #     od['01'] = nn.PReLU()
    #     od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['11'] = nn.PReLU()
    #     od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['21'] = nn.PReLU()
    #     cm = ConcatModule(od)
    #     sod['0'] = cm
    #
    #     # 2
    #     od = OrderedDict()
    #     od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['01'] = nn.PReLU()
    #     od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['11'] = nn.PReLU()
    #     od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['21'] = nn.PReLU()
    #     cm = ConcatModule(od)
    #     sod['1'] = cm
    #
    #     # 3
    #     od = OrderedDict()
    #     od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['01'] = nn.PReLU()
    #     od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['11'] = nn.PReLU()
    #     od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['21'] = nn.PReLU()
    #     cm = ConcatModule(od)
    #     sod['2'] = cm
    #
    #     # 4
    #     od = OrderedDict()
    #     od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['01'] = nn.PReLU()
    #     od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['11'] = nn.PReLU()
    #     od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['21'] = nn.PReLU()
    #     cm = ConcatModule(od)
    #     sod['3'] = cm
    #
    #     # 5
    #     od = OrderedDict()
    #     od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
    #     od['01'] = nn.PReLU()
    #     od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['11'] = nn.PReLU()
    #     od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
    #     od['21'] = nn.PReLU()
    #     cm = ConcatModule(od)
    #     sod['4'] = cm
    #
    #     sod['5'] =nn.Conv2d(blockinputdim * 3, penultimatelayerdim, 1, padding=0)
    #     sod['6'] = nn.PReLU()
    #     sod['7'] = nn.Conv2d(penultimatelayerdim, 26, 1, padding=0)
    #
    #     return nn.Sequential(sod)









