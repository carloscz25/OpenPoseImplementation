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

batchsize = 8
paths = {}
paths['local'] = ['/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017','/home/carlos/PycharmProjects/PublicDatasets/MPII/images']
paths['cloud'] = ['/mnt/disks/sdb/datasets/coco/train2017','/mnt/disks/sdb/datasets/mpii/images']

dataset = OpenPoseDataset(['coco','mpii'], [0.9,0.1], paths['cloud'], ['train.json', 'trainmpii.json'])
dataloader = torch.utils.data.DataLoader(dataset, batchsize, collate_fn=collatefn)
singlebatch = None
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

    print(step)
    print('L', overallLossL)
    print('S', overallLossS)

    if (step % 50)==0:
        if step > 0:
            torch.save(model.state_dict(), 'checkpoints/current.chp')

    #statistics form trainin monitorization

    #main
    # scs={}
    # scs['L'] = overallLossL
    # scs['S'] = overallLossSd
    # writer.add_scalars('main',scs, step)
    writer.add_scalar('L', overallLossL, step)
    writer.add_scalar('S', overallLossS, step)

    #checking memory
    # mem = psutil.virtual_memory()
    # writer.add_scalar('used', mem.used, step)
    # writer.add_scalar('free', mem.free, step)

    #adding images
    if step % 1000 == 0:
        if step > 0:
            for i in range(batchsize):
                original_image = cv2.imread(image_url[i])
                im1 = getimagewithpartheatmaps(original_image, S[i])
                im1t = torch.from_numpy(im1)
                writer.add_image('im'+str(i), im1t, step, dataformats='HWC')

    # monitor_gradient_mean(writer, mods, ['L1','L2','L3','L4','L5','S1','S1'], step)
    # monitor_parameter_mean(writer, mods, ['L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S1'], step)

    # monitor_gradient_shift(mods, ['L3','L4', 'L5','S1','S2'], step, 'gradientsmonitoring.grad')


    # add_scalars_for_modname(mods, '_F_', step)
    # add_scalars_for_modname(mods, '_cpm', step)
    # add_scalars_for_modname(mods, '_L1_', step)
    # add_scalars_for_modname(mods, '_L2_', step)
    # add_scalars_for_modname(mods, '_L3_', step)
    # add_scalars_for_modname(mods, '_L4_', step)
    # add_scalars_for_modname(mods, '_L5_', step)
    # add_scalars_for_modname(mods, '_S1_', step)
    # add_scalars_for_modname(mods, '_S2_', step)





    #monitorizando provisionalmente
    # annadj = adjustannotationpoints(ann[0], (28,28), original_image_dim[0])
    # imres = getimagewithdisplayedannotations(cv2.imread(image_url[0]), annadj)
    # cv2.imshow('w', imres)
    # cv2.waitKey(0)

    # im = drawSvsStarget(S, Starget)
    # cv2.imshow('w', im)
    # cv2.waitKey(0)













