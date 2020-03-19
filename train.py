from dataloader import *
from helpers import *
from model import *


datasetnames = ['coco', 'mpii']

model = OpenPoseModel()
model.train()

paramsF = model.F.parameters()
paramsL = model.L.parameters()
paramsS = model.S.parameters()

optimizerF = torch.optim.Adam(paramsF, lr=0.001)
optimizerL = torch.optim.Adam(paramsL, lr=0.001)
optimizerS = torch.optim.Adam(paramsS, lr=0.001)

criterionF = torch.nn.MSELoss('sum')
criterionL = torch.nn.MSELoss('sum')
criterionS = torch.nn.MSELoss('sum')


dataloader = OpenPoseDataset(['coco','mpii'], [0.9,0.1], ['/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017','/home/carlos/PycharmProjects/PublicDatasets/MPII/images'], ['train.json', 'trainmpii.json'])
while(True):
    batch = next(dataloader)
    im = batch[1]
    Starget = batch[5]
    Ltarget = batch[6]
    im = im.unsqueeze(0)
    F = model.F(im)
    #run L stages
    L = F
    for i in range(4):
        L, Lflat = model.forward_L(L, i)
        if (i<3):
            L = torch.cat((F, L), 1)
    Lflat = model.Ldecoder(Lflat)
    L = Lflat.reshape((1,2,112,112))
    #run S stages
    S, Sflat = model.forward_S()

    # before loss and the backward pass, need first set to 0 all those pixels from S and L where in Starget and Ltarget are zero
    # to avoid penalizing unnanotated pixels
    L[Ltarget == 0] = 0
    S[Starget == 0] = 0

    lossF = criterionF()#pendiente
    lossL = criterionL(L, Ltarget)
    lossS = criterionS(S, Starget)

    model.zero_grad()

    optimizerF.step()
    optimizerL.step()
    optimizerS.step()

    overallLoss = lossF + lossL + lossS

    print('lossF', lossF)
    print('lossL', lossL)
    print('lossS', lossS)







