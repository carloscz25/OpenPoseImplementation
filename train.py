from openposedataset import *
from helpers import *
from model import *
import time
from  torch.utils.tensorboard import SummaryWriter
from monitoring import *
import pickle
import random
# import psutil

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



writer = SummaryWriter()

datasetnames = ['coco', 'mpii']

def set_requires_grad(model, truefalse):
    for p in model.parameters():
        p.requires_grad = truefalse






def add_scalar_max_min(name, step, arr):
    d = {}
    d['max'] = torch.max(arr).item()
    d['min'] = torch.min(arr).item()
    writer.add_scalars(name, d, step)

model = OpenPoseModel()
model = model.to(device)

#monitoring part
mods = {}
mods2 = {}
allmodules(model, mods)
for m in mods.keys():
    if len(mods[m]._parameters)>0:
        mods2[m] = mods[m]
mods = mods2
mods2 = None


if os.path.exists('checkpoints/current.chp'):
    sd = torch.load(open('checkpoints/current.chp', 'rb'))
    model.load_state_dict(sd)
else:
    pass
model.train()

set_requires_grad(model.F, False)
set_requires_grad(model.cpm1, False)
set_requires_grad(model.cpm1prlu, False)
set_requires_grad(model.cpm2, False)
set_requires_grad(model.cpm2prlu, False)
set_requires_grad(model.L1, False)
set_requires_grad(model.L2, False)
set_requires_grad(model.L3, False)
set_requires_grad(model.L4, False)
set_requires_grad(model.L5, False)
set_requires_grad(model.S1, False)
set_requires_grad(model.S2, False)

learningrate = 0.0001
#loss functions and optimizers
criterionL1 = torch.nn.MSELoss('none')
optimizerL1 = torch.optim.Adam(list(model.L1.parameters()) + list(model.cpm1.parameters()) + list(model.cpm1prlu.parameters()) + list(model.cpm2.parameters()) + list(model.cpm2prlu.parameters()), lr=learningrate)
# optimizerL1 = torch.optim.Adam(list(model.L1.parameters()), lr=learningrate)
criterionL2 = torch.nn.MSELoss('none')
optimizerL2 = torch.optim.Adam(list(model.L2.parameters()), lr=learningrate)
criterionL3 = torch.nn.MSELoss('none')
optimizerL3 = torch.optim.Adam(list(model.L3.parameters()), lr=learningrate)
criterionL4 = torch.nn.MSELoss('none')
optimizerL4 = torch.optim.Adam(list(model.L4.parameters()), lr=learningrate)
criterionL5 = torch.nn.MSELoss('none')
optimizerL5 = torch.optim.Adam(list(model.L5.parameters()), lr=learningrate)
criterionS1 = torch.nn.MSELoss('none')
optimizerS1 = torch.optim.Adam(list(model.S1.parameters()), lr=learningrate)
criterionS2 = torch.nn.MSELoss('none')
optimizerS2 = torch.optim.Adam(list(model.S2.parameters()), lr=learningrate)


def collatefn(o):

    oa = []
    ext = []
    for r in range(len(o[0])):
        l = []
        for t in o:
           l.append(t[r])
        if r in (1,4):
            ext.append(l)#dicts not allowed to be stacked as tensor
        else:
            oa.append(torch.stack(l,0))
    return oa, ext

batchsize = 16
epochs = 10
paths = {}
paths['local'] = ['/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017','/home/carlos/PycharmProjects/PublicDatasets/MPII/images']
paths['cloud'] = ['/mnt/disks/sdb/datasets/coco/train2017','/mnt/disks/sdb/datasets/mpii/images']

dataset = OpenPoseDataset(['coco','mpii'], [0.9,0.1], paths['cloud'], ['traincoco.json', 'trainmpii.json'])
dataloader = torch.utils.data.DataLoader(dataset, batchsize, collate_fn=collatefn)
singlebatch = None
for i in range(epochs):
    print(i)
    for step, batch in enumerate(dataloader):
        # if singlebatch==None:
        ([impreprocessed, Starget, Ltarget, original_image_dim], [ann, image_url]) = batch

        Ltarget = Ltarget.float()
        Starget = Starget.float()

        Ltarget = Ltarget.to(device)
        Starget = Starget.to(device)
        impreprocessed = impreprocessed.to(device)


        F = model.F(impreprocessed)

        # add_scalar_max_min('F', step, F)


        #CPM part
        set_requires_grad(model.cpm1, True)
        set_requires_grad(model.cpm1prlu, True)
        set_requires_grad(model.cpm2, True)
        set_requires_grad(model.cpm2prlu, True)
        set_requires_grad(model.L1, True)
        F = model.cpm1(F)
        F = model.cpm1prlu(F)
        F = model.cpm2(F)
        F = model.cpm2prlu(F)
        F = F.detach()

        nsubruns=1
        #run L stages
        #stage1
        for i in range(nsubruns):
            model.L1.zero_grad()
            L = model.L1(F)
            L[Ltarget==0] = 0
            lossL1 = criterionL1(L, Ltarget)
            lossL1.backward()
            optimizerL1.step()
        L = model.L1(F)
        L = L.detach()
        set_requires_grad(model.cpm1, False)
        set_requires_grad(model.cpm1prlu, False)
        set_requires_grad(model.cpm2, False)
        set_requires_grad(model.cpm2prlu, False)
        set_requires_grad(model.L1, False)

        set_requires_grad(model.L2, True)
        for i in range(nsubruns):
            model.L2.zero_grad()
            Linput = torch.cat((F, L), 1)
            L = model.L2(Linput)
            L[Ltarget == 0] = 0
            lossL2 = criterionL2(L, Ltarget)
            lossL2.backward()
            optimizerL2.step()

        Linput = torch.cat((F, L), 1)
        L = model.L2(Linput)
        L = L.detach()
        set_requires_grad(model.L2, False)

        set_requires_grad(model.L3, True)
        for i in range(nsubruns):
            model.L3.zero_grad()
            Linput = torch.cat((F, L), 1)
            L = model.L3(Linput)
            L[Ltarget == 0] = 0
            lossL3 = criterionL3(L, Ltarget)
            lossL3.backward()
            optimizerL3.step()
        Linput = torch.cat((F, L), 1)
        L = model.L3(Linput)
        L = L.detach()
        set_requires_grad(model.L3, False)

        set_requires_grad(model.L4, True)
        for i in range(nsubruns):
            Linput = torch.cat((F, L), 1)
            L = model.L4(Linput)
            L[Ltarget == 0] = 0
            lossL4 = criterionL4(L, Ltarget)
            lossL4.backward()
            optimizerL4.step()
            model.L4.zero_grad()
        Linput = torch.cat((F, L), 1)
        L = model.L4(Linput)
        L = L.detach()
        set_requires_grad(model.L4, False)

        set_requires_grad(model.L5, True)
        for i in range(nsubruns):
            Linput = torch.cat((F, L), 1)
            L = model.L5(Linput)
            L[Ltarget == 0] = 0
            lossL5 = criterionL5(L, Ltarget)
            lossL5.backward()
            optimizerL5.step()
            model.L5.zero_grad()
        Linput = torch.cat((F, L), 1)
        L = model.L5(Linput)
        L = L.detach()
        set_requires_grad(model.L5, False)

        # S stages
        # stage1
        set_requires_grad(model.S1, True)
        for i in range(nsubruns):
            Sinput = torch.cat((F, L), 1)
            S = model.S1(Sinput)
            S[Starget == 0] = 0
            lossS1 = criterionS1(S, Starget)
            lossS1.backward()
            optimizerS1.step()
            model.S1.zero_grad()
        Sinput = torch.cat((F, L), 1)
        S = model.S1(Sinput)
        S = S.detach()
        set_requires_grad(model.S1, False)

        #stage2
        set_requires_grad(model.S2, True)
        for i in range(nsubruns):
            Sinput = torch.cat((F, L, S), 1)
            S = model.S2(Sinput)
            S[Starget == 0] = 0
            lossS2 = criterionS1(S, Starget)
            if i < (nsubruns-1):
                lossS2.backward()
            else:
                lossS2.backward()
            optimizerS2.step()
            model.S2.zero_grad()
        Sinput = torch.cat((F, L, S), 1)
        S = model.S2(Sinput)
        S = S.detach()
        set_requires_grad(model.S2, False)

        # S, L, F = None, None, None



        model.zero_grad()

        overallLossL =  lossL1 + lossL2 + lossL3 + lossL4 + lossL5
        overallLossS =  lossS1 + lossS2

        print('epoch' + str(i) + 'step' + str(step))
        print('L', overallLossL)
        print('S', overallLossS)

        if (step % 50)==0:
            if step > 0:
                torch.save(model.state_dict(), 'checkpoints/current.chp')

        #statistics form trainin monitorization


        writer.add_scalar('L'+str(i), overallLossL, step)
        writer.add_scalar('S'+str(i), overallLossS, step)


        #adding images
        if step % 100 == 0:
            if step > 0:
                for i in range(batchsize):
                    original_image = cv2.imread(image_url[i])
                    im1 = getimagewithpartheatmaps(original_image, S[i])
                    im1t = torch.from_numpy(im1)
                    writer.add_image('im'+str(i), im1t, step, dataformats='HWC')















