import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.transforms import *

class ConcatModule(nn.Module):
    def __init__(self, od):
        super(ConcatModule, self).__init__()
        self.ordereddict = od

    def forward(self, input):
        l = []
        for i, t in enumerate(self.ordereddict):
            acc = None
            if i == 0:
                acc = input
            acc = t.forward(acc)



class OpenPoseModel(nn.Module):

    def __init__(self):
        super(OpenPoseModel, self).__init__()
        self.F = self.__init_features_layerset()
        #LStages
        self.L1 = self.__init_Lstage(128, 96, 256)
        # self.L1toL2 = ConcatModule(OrderedDict((('0',self.F), ('1',self.L1))))
        self.L2 = self.__init_Lstage(180, 128, 512)
        # self.L2toL3 = ConcatModule(OrderedDict((('0',self.F), ('1',self.L2))))
        self.L3 = self.__init_Lstage(180, 128, 512)
        # self.L3toL4 = ConcatModule(OrderedDict((self.F, self.L3)))
        self.L4 = self.__init_Lstage(180, 128, 512)
        self.Ldecoder = nn.Linear(52*28*28,224**2)

        # self.LtoS = ConcatModule(OrderedDict((self.L4, self.F)))

        #SStages
        self.S1 = self.__init_Sstage(180, 96, 256)
        # self.S1toS2 = ConcatModule(OrderedDict((self.S1,self.L4, self.F)))
        self.S2 = self.__init_Sstage(206, 128, 512)
        self.Sdecoder = nn.Linear(26 * 28 * 28, 224 ** 2)

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


    def __init_Lstage(self, stageinputdim=128, blockinputdim=96, penultimatelayerdim=256):#inputdim is 128 at first go, and 288 at the rest (96*3)

        lod = OrderedDict()

        #1
        od = OrderedDict()
        od['01'] = nn.Conv2d(stageinputdim,blockinputdim,3,padding=1)
        od['011'] = nn.PReLU()
        od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['021'] = nn.PReLU()
        od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['031'] = nn.PReLU()

        lod['0'] = nn.Sequential(od)

        #2
        od = OrderedDict()
        od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['011'] = nn.PReLU()
        od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['021'] = nn.PReLU()
        od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['031'] = nn.PReLU()

        lod['1'] = nn.Sequential(od)

        # 3
        od = OrderedDict()
        od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['011'] = nn.PReLU()
        od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['021'] = nn.PReLU()
        od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['031'] = nn.PReLU()

        lod['2'] = nn.Sequential(od)

        # 4
        od = OrderedDict()
        od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['011'] = nn.PReLU()
        od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['021'] = nn.PReLU()
        od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['031'] = nn.PReLU()

        lod['3'] = nn.Sequential(od)

        # 5
        od = OrderedDict()
        od['01'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['011'] = nn.PReLU()
        od['02'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['021'] = nn.PReLU()
        od['03'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['031'] = nn.PReLU()

        lod['4'] = nn.Sequential(od)

        lod['5'] = nn.Conv2d(blockinputdim*3, penultimatelayerdim, 1, padding=0)
        lod['6'] = nn.PReLU()
        lod['7'] = nn.Conv2d(penultimatelayerdim, 52, 1, padding=0)

        return nn.Sequential(lod)


    def __init_Sstage(self, stageinputdim=128, blockinputdim=96, penultimatelayerdim=256):  # inputdim is 128 at first go, and 288 at the rest (96*3)
        sod = OrderedDict()
        #1
        od = OrderedDict()
        od['0'] = nn.Conv2d(stageinputdim, blockinputdim, 3, padding=1)
        od['01'] = nn.PReLU()
        od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['11'] = nn.PReLU()
        od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['21'] = nn.PReLU()
        cm = ConcatModule(od)
        sod['0'] = cm

        # 2
        od = OrderedDict()
        od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['01'] = nn.PReLU()
        od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['11'] = nn.PReLU()
        od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['21'] = nn.PReLU()
        cm = ConcatModule(od)
        sod['1'] = cm

        # 3
        od = OrderedDict()
        od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['01'] = nn.PReLU()
        od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['11'] = nn.PReLU()
        od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['21'] = nn.PReLU()
        cm = ConcatModule(od)
        sod['2'] = cm

        # 4
        od = OrderedDict()
        od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['01'] = nn.PReLU()
        od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['11'] = nn.PReLU()
        od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['21'] = nn.PReLU()
        cm = ConcatModule(od)
        sod['3'] = cm

        # 5
        od = OrderedDict()
        od['0'] = nn.Conv2d(blockinputdim*3, blockinputdim, 3, padding=1)
        od['01'] = nn.PReLU()
        od['1'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['11'] = nn.PReLU()
        od['2'] = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1)
        od['21'] = nn.PReLU()
        cm = ConcatModule(od)
        sod['4'] = cm

        sod['5'] =nn.Conv2d(blockinputdim * 3, penultimatelayerdim, 1, padding=0)
        sod['6'] = nn.PReLU()
        sod['7'] = nn.Conv2d(penultimatelayerdim, 26, 1, padding=0)

        return nn.Sequential(sod)





    def forward_L(self, features):

        #start L stages : 4 stages
        #stage1
        Ls = self.L1(features)
        # Ls = self.L1toL2()
        Ls = torch.cat(Ls, features,1)
        Ls = self.L2(Ls)
        # Ls = self.L2toL3()
        Ls = torch.cat(Ls, features, 1)
        Ls = self.L3(Ls)
        # Ls = self.L3toL4()
        torch.cat(Ls, features, 1)
        Ls = self.L4(Ls)
        flattened = torch.flatten(Ls)
        return Ls, flattened

    def forward_S(self, L4, features):

        # Ss = self.LtoS()
        Ss = torch.cat((L4, features), 1)
        Ss = self.S1(Ss)
        Ss = torch.cat((Ss, L4, features), 1)
        # Ss = self.S1toS2()
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











