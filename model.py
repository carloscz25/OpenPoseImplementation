import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.transforms import *




class LStageBlock(nn.Module):
    def __init__(self,firstlayerinputdim=128, blockinputdim=96):
        super(LStageBlock, self).__init__()
        self.c2d1 = nn.Conv2d(firstlayerinputdim, blockinputdim, 3, padding=1, bias=True)
        self.prl1 = nn.PReLU()
        self.c2d2 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1, bias=True)
        self.prl2 = nn.PReLU()
        self.c2d3 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1,bias=True)
        self.prl3 = nn.PReLU()
    def forward(self, input):
        o1 = self.prl1(self.c2d1(input))
        o2 = self.prl2(self.c2d2(o1))
        o3 = self.prl3(self.c2d3(o2))
        o = torch.cat((o1,o2,o3),1)
        return o

class LStage(nn.Module):
    def __init__(self, n_limbs, stageinputdim=128, blockinputdim=96, penultimatelayerdim=256):
        super(LStage, self).__init__()
        self.b1 = LStageBlock(stageinputdim, blockinputdim)
        self.b2 = LStageBlock(blockinputdim*3, blockinputdim)
        self.b3 = LStageBlock(blockinputdim*3, blockinputdim)
        self.b4 = LStageBlock(blockinputdim*3, blockinputdim)
        self.b5 = LStageBlock(blockinputdim*3, blockinputdim)
        self.penultimatelayer = nn.Conv2d(blockinputdim*3, penultimatelayerdim, 1, padding=0, bias=True)
        self.prlpl = nn.PReLU()
        self.ouputlayer = nn.Conv2d(penultimatelayerdim, (n_limbs*2), 1, padding=0, bias=True)

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
    def __init__(self,firstlayerinputdim=206, blockinputdim=96):
        super(SStageBlock, self).__init__()
        self.c2d1 = nn.Conv2d(firstlayerinputdim, blockinputdim, 3, padding=1, bias=True)
        self.prl1 = nn.PReLU()
        self.c2d2 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1, bias=True)
        self.prl2 = nn.PReLU()
        self.c2d3 = nn.Conv2d(blockinputdim, blockinputdim, 3, padding=1, bias=True)
        self.prl3 = nn.PReLU()
    def forward(self, input):
        o1 = self.prl1(self.c2d1(input))
        o2 = self.prl2(self.c2d2(o1))
        o3 = self.prl3(self.c2d3(o2))
        o = torch.cat((o1,o2,o3),1)
        return o

class SStage(nn.Module):
    def __init__(self, n_parts, stageinputdim=206, blockinputdim=96, penultimatelayerdim=256):
        super(SStage, self).__init__()
        self.b1 = SStageBlock(stageinputdim, blockinputdim)
        self.b2 = SStageBlock(blockinputdim*3, blockinputdim)
        self.b3 = SStageBlock(blockinputdim*3, blockinputdim)
        self.b4 = SStageBlock(blockinputdim*3, blockinputdim)
        self.b5 = SStageBlock(blockinputdim*3, blockinputdim)
        self.penultimatelayer = nn.Conv2d(blockinputdim*3, penultimatelayerdim, 1, padding=0, bias=True)
        self.prlpl = nn.PReLU()
        self.ouputlayer = nn.Conv2d(penultimatelayerdim, n_parts, 1, padding=0, bias=True)

    def forward(self, input):
        o = self.b1(input)
        o = self.b2(o)
        o = self.b3(o)
        o = self.b4(o)
        o = self.b5(o)
        o = self.prlpl(self.penultimatelayer(o))
        o = self.ouputlayer(o)
        return o





class OpenPoseModel(nn.Module):

    def __init__(self):
        super(OpenPoseModel, self).__init__()
        self.F = self.__initvgg19()
        #CPM
        self.cpm1 = nn.Conv2d(512, 256, 3, padding=1, bias=True)
        self.cpm1prlu = nn.PReLU()
        self.cpm2 = nn.Conv2d(256, 128, 3, padding=1, bias=True)
        self.cpm2prlu = nn.PReLU()
        #LStages
        self.L1 = LStage(10, 128, 96, 256) #128, 96 ,256
        self.L2 = LStage(10, 148, 128, 512)  # 128, 96 ,256
        self.L3 = LStage(10, 148, 128, 512)  # 128, 96 ,256
        self.L4 = LStage(10, 148, 128, 512)  # 128, 96 ,256
        self.L5 = LStage(10, 148, 128, 512)  # 128, 96 ,256

        #SStages
        self.S1 = SStage(12, 148, 96, 256)#180, 96, 256
        self.S2 = SStage(12, 160, 128, 512)  # 180, 96, 256
        # self.S3 = SStage(12, 160, 128, 512)  # 180, 96, 256
        # self.S4 = SStage(12, 160, 128, 512)  # 180, 96, 256

    def init_weights(self, fn):
        self.cpm1.apply(fn)
        self.cpm1prlu.apply(fn)
        self.cpm2.apply(fn)
        self.cpm2prlu.apply(fn)
        # LStages
        self.L1.apply(fn)
        self.L2.apply(fn)
        self.L3.apply(fn)
        self.L4.apply(fn)
        self.L5.apply(fn)

        # SStages
        self.S1.apply(fn)
        self.S2.apply(fn)
        #extension
        # self.S3.apply(fn)
        # self.S4.apply(fn)



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

    def __initvgg19(self):
        from  torchvision.models import vgg19_bn
        m = vgg19_bn(pretrained=True)
        # m.load_state_dict(torch.load('vgg19bn.pth'))
        vgg19dict = OrderedDict()
        for i, mod in enumerate(m.features._modules):
            if i == 33:
                break
            vgg19dict[mod] = m.features._modules[mod]

        return torch.nn.Sequential(vgg19dict)














