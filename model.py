import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict


class OpenPoseModel(nn.Module):

    vggmodules = None


    def __init__(self):
        #adding first 10 layers of VGG-19 net (pretrained with the downloaded state dict)
        self.vggmodules = nn.Sequential(self.__preparevggmodules__())
        self.LModule = nn.Sequential(OrderedDict([self.__newconvblock(0),self.__newconvblock(1),self.__newconvblock(2)]))
        self.SModule = nn.Sequential()



    def __preparevggmodules__(self):
        od = OrderedDict()
        model = models.vgg19_bn(pretrained=False)
        model.load_state_dict(torch.load('vgg19bn.pth'))
        for i in model._modules['features']._modules.keys():
            # se deben extraer modulos desde el indice 0 al 32
            if (int(i)<33):
                od[i] = model._modules['features']._modules[i]
        return od

    def __newconvblock(self, index):
        seq = nn.Sequential()
        seq.add_module(str(index) + 'c1', nn.Conv2d(512, 512, 3, 1, 0))
        seq.add_module(str(index) + 'c2', nn.Conv2d(512, 512, 3, 1, 0))
        seq.add_module(str(index) + 'c3', nn.Conv2d(512, 512, 3, 1, 0))
        return seq

    def forward(self, features):
        pass




y = 2