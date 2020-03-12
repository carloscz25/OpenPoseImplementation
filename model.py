import torchvision.models as models
import torch


#instantiating the vgg-19 model
model = models.vgg19_bn(pretrained=False)
model.load_state_dict(torch.load('vgg19bn.pth'))
model._modules['features']._modules['0']
y = 2