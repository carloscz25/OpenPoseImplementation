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

use_unlabelled_mask = False

writer = SummaryWriter()

datasetnames = ['coco', 'mpii']

def set_requires_grad(model, truefalse):
    for p in model.parameters():
        p.requires_grad = truefalse

#avoid non labelled data to penalize positives
model = OpenPoseModel()
model = model.to(device)

#preloading for monitoring
d = {}
allmodules(model, d, '')

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
# set_requires_grad(model.S3, False)
# set_requires_grad(model.S4, False)

learningrate = 0.00001
learningrate = 1e-4
#loss functions and optimizers
criterionL1 = torch.nn.MSELoss('none')
optimizerL1 = torch.optim.Adam(list(model.L1.parameters()) + list(model.cpm1.parameters()) + list(model.cpm1prlu.parameters()) + list(model.cpm2.parameters()) + list(model.cpm2prlu.parameters()) + list(model.F.parameters()), lr=learningrate)
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
# criterionS3 = torch.nn.MSELoss('none')
# optimizerS3 = torch.optim.Adam(list(model.S3.parameters()), lr=learningrate)
# criterionS4 = torch.nn.MSELoss('none')
# optimizerS4 = torch.optim.Adam(list(model.S4.parameters()), lr=learningrate)

if os.path.exists('checkpoints/optimizers.chp'):
    sd = pickle.load(open('checkpoints/optimizers.chp', 'rb'))
    optimizerL1.load_state_dict(sd['L1'])
    optimizerL2.load_state_dict(sd['L2'])
    optimizerL3.load_state_dict(sd['L3'])
    optimizerL4.load_state_dict(sd['L4'])
    optimizerL5.load_state_dict(sd['L5'])
    optimizerS1.load_state_dict(sd['S1'])
    optimizerS2.load_state_dict(sd['S2'])
    # optimizerS3.load_state_dict(sd['S3'])
    # optimizerS4.load_state_dict(sd['S4'])
else:
    pass



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

batchsize = 4
epochs = 10
paths = {}
paths['local'] = ['/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017','/home/carlos/PycharmProjects/PublicDatasets/MPII/images']
paths['cloud'] = ['/mnt/disks/sdb/datasets/coco/train2017','/mnt/disks/sdb/datasets/mpii/images']

dataset = OpenPoseDataset(['coco','mpii'], [0.9,0.1], paths['local'], ['traincoco.json', 'trainmpii.json'])
dataloader = torch.utils.data.DataLoader(dataset, batchsize, collate_fn=collatefn)
singlebatch = None
for i in range(epochs):
    print(i)
    for step, batch in enumerate(dataloader):
        # if singlebatch==None:
        #     singlebatch = batch
        ([impreprocessed, Starget, Ltarget, original_image_dim], [ann, image_url]) = batch

        Ltarget = Ltarget.float()
        Starget = Starget.float()

        Ltarget = Ltarget.to(device)
        Starget = Starget.to(device)
        impreprocessed = impreprocessed.to(device)


        F = model.F(impreprocessed)

        #CPM part
        set_requires_grad(model.F, True)
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
            model.F.zero_grad()
            model.cpm1.zero_grad()
            model.cpm1prlu.zero_grad()
            model.cpm2.zero_grad()
            model.cpm2prlu.zero_grad()
            model.L1.zero_grad()
            L = model.L1(F)
            if use_unlabelled_mask == True:
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
            if use_unlabelled_mask == True:
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
            if use_unlabelled_mask == True:
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
            if use_unlabelled_mask == True:
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
            if use_unlabelled_mask == True:
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
            if use_unlabelled_mask == True:
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
            if use_unlabelled_mask == True:
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

        #stage3
        # set_requires_grad(model.S3, True)
        # for i in range(nsubruns):
        #     Sinput = torch.cat((F, L, S), 1)
        #     S = model.S3(Sinput)
        #     if use_unlabelled_mask == True:
        #         S[Starget == 0] = 0
        #     lossS3 = criterionS3(S, Starget)
        #     if i < (nsubruns - 1):
        #         lossS3.backward()
        #     else:
        #         lossS3.backward()
        #     optimizerS3.step()
        #     model.S3.zero_grad()
        # Sinput = torch.cat((F, L, S), 1)
        # S = model.S3(Sinput)
        # S = S.detach()
        # set_requires_grad(model.S3, False)

        # stage4
        # set_requires_grad(model.S4, True)
        # for i in range(nsubruns):
        #     Sinput = torch.cat((F, L, S), 1)
        #     S = model.S4(Sinput)
        #     if use_unlabelled_mask == True:
        #         S[Starget == 0] = 0
        #     lossS4 = criterionS4(S, Starget)
        #     if i < (nsubruns - 1):
        #         lossS4.backward()
        #     else:
        #         lossS4.backward()
        #     optimizerS4.step()
        #     model.S4.zero_grad()
        # Sinput = torch.cat((F, L, S), 1)
        # S = model.S4(Sinput)
        # S = S.detach()
        # set_requires_grad(model.S4, False)

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
                optimizers = {}
                optimizers['L1'] = optimizerL1.state_dict()
                optimizers['L2'] = optimizerL2.state_dict()
                optimizers['L3'] = optimizerL3.state_dict()
                optimizers['L4'] = optimizerL4.state_dict()
                optimizers['L5'] = optimizerL5.state_dict()
                optimizers['S1'] = optimizerS1.state_dict()
                optimizers['S2'] = optimizerS2.state_dict()
                pickle.dump(optimizers, open('checkpoints/optimizers.chp','wb'))


        #statistics form trainin monitorization


        writer.add_scalar('L'+str(i), overallLossL.item(), step)
        writer.add_scalar('S'+str(i), overallLossS.item(), step)

        #monitoring gradients
        # monitor_gradient_shift(writer, d, ("L","S"), step, 'gradientshift.grad')


        #adding images
        if step % 100 == 0:
            if step >= 0:
                for i in range(batchsize):
                    original_image = cv2.imread(image_url[i])
                    try:
                        im1 = getimagewithpartheatmaps(original_image, S[i])
                        # cv2.imshow('w', im1)
                        # cv2.waitKey(0)
                        im1t = torch.from_numpy(im1)
                        writer.add_image('im'+str(i), im1t, step, dataformats='HWC')
                    except:
                        pass














