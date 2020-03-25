from openposedataset import *
from helpers import *
from model import *
import time
from  torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

datasetnames = ['coco', 'mpii']

def initw(o):
    #weight initialization
    if o.__class__.__name__=='ReLU':
        return
    if o.__class__.__name__=='MaxPool2d':
        return
    if o.__class__.__name__=='Sequential':
        return
    if o.__class__.__name__=='LStageBlock':
        return
    if o.__class__.__name__=='SStageBlock':
        return
    if o.__class__.__name__=='LStage':
        return
    if o.__class__.__name__=='SStage':
        return
    o.weight.data.normal_(0.0, 0.1)
    if o.__class__.__name__=='PReLU':
        return
    o.bias.data.normal_(0.0, 0.1)
    y=2

def allmodules(model, d, currname):
    for name, module in model._modules.items():
        d[currname + "_" + name] = module
        allmodules(module, d, currname + "_" + name)

def add_scalars_for_modname(mods, name, step):
    d = {}
    for m in mods.keys():
        try:
            if m.index(name) >= 0:
                mod = mods[m]
                minp, maxp = torch.min(mod._parameters['weight']), torch.max(mod._parameters['weight'])
                d[m+'_minP_'] = minp.item()
                d[m+'_maxP_'] = maxp.item()
                minb, maxb = torch.min(mod._parameters['bias']), torch.max(mod._parameters['bias'])
                d[m + '_minB_'] = minb.item()
                d[m + '_maxB_'] = maxb.item()
        except:
            pass
    writer.add_scalars(name, d, step)


def add_scalar_max_min(name, step, arr):
    d = {}
    d['max'] = torch.max(arr).item()
    d['min'] = torch.min(arr).item()
    writer.add_scalars(name, d, step)

model = OpenPoseModel(224,224,224)

#monitoring part
mods = {}
mods2 = {}
allmodules(model, mods, '')
for m in mods.keys():
    if len(mods[m]._parameters)>0:
        mods2[m] = mods[m]
mods = mods2
mods2 = None

# model.init_weights(initw)
paramsL = list(model.L1.parameters()) + list(model.L2.parameters()) + list(model.L3.parameters()) + list(model.L4.parameters()) + list(model.L5.parameters()) + list(model.cpm1.parameters()) + list(model.cpm1prlu.parameters()) +list(model.cpm2.parameters()) + list(model.cpm2prlu.parameters())
paramsS = list(model.S1.parameters()) + list(model.S2.parameters())

if os.path.exists('checkpoints/current.chp'):
    sd = torch.load(open('checkpoints/current.chp', 'rb'))
    model.load_state_dict(sd)
else:
    pass
model.train()



optimizerL = torch.optim.Adam(paramsL, lr=0.00001)
optimizerS = torch.optim.Adam(paramsS, lr=0.00001)


criterionF = torch.nn.MSELoss('none')
criterionL = torch.nn.MSELoss('none')
criterionS = torch.nn.MSELoss('none')

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
paths = {}
paths['local'] = ['/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017','/home/carlos/PycharmProjects/PublicDatasets/MPII/images']
paths['cloud'] = ['/mnt/disks/sdb/datasets/coco/train2017','/mnt/disks/sdb/datasets/mpii/images']

dataset = OpenPoseDataset(['coco','mpii'], [0.9,0.1], paths['local'], ['train.json', 'trainmpii.json'])
dataloader = torch.utils.data.DataLoader(dataset, batchsize, collate_fn=collatefn)
for step, batch in enumerate(dataloader):

    ([impreprocessed, Starget, Ltarget, original_image_dim], [ann, image_url]) = batch


    Ltarget = Ltarget.float()
    Starget = Starget.float()
    # Ltarget = torch.from_numpy(Ltarget).float().unsqueeze(0)
    # Starget = torch.from_numpy(Starget).float().unsqueeze(0)

    # im = im.unsqueeze(0)
    add_scalar_max_min('impreprocessed', step, impreprocessed)
    F = model.F(impreprocessed)
    add_scalar_max_min('F', step, F)

    #CPM part
    F = model.cpm1(F)
    F = model.cpm1prlu(F)
    F = model.cpm2(F)
    F = model.cpm2prlu(F)
    add_scalar_max_min('CPM', step, F)

    #run L stages
    #stage1

    L = model.L1(F)
    add_scalar_max_min('L1', step, L)
    Linput = torch.cat((F, L), 1)
    L = model.L2(Linput)
    add_scalar_max_min('L2', step, L)
    Linput = torch.cat((F, L), 1)
    L = model.L3(Linput)
    add_scalar_max_min('L3', step, L)
    Linput = torch.cat((F, L), 1)
    L = model.L4(Linput)
    add_scalar_max_min('L4', step, L)
    Linput = torch.cat((F, L), 1)
    L = model.L5(Linput)
    add_scalar_max_min('L5', step, L)
    # S stages
    # stage1
    Sinput = torch.cat((F, L), 1)
    S = model.S1(Sinput)
    add_scalar_max_min('S1', step, L)
    Sinput = torch.cat((F, L, S), 1)
    S = model.S2(Sinput)
    add_scalar_max_min('S2', step, L)

    model.zero_grad()
    # L[Ltarget==0]=0
    # S[Starget == 0] = 0
    lossL = criterionL(L, Ltarget)
    lossS = criterionS(S, Starget)
    lossL.backward(retain_graph=True)
    lossS.backward()
    optimizerL.step()
    optimizerS.step()


    overallLoss =  lossL + lossS
    print(step)
    print('L', lossL)
    print('S', lossS)

    if (step % 100)==0:
        torch.save(model.state_dict(), 'checkpoints/current.chp')

    #statistics form trainin monitorization

    #main
    scs={}
    scs['L'] = lossL
    scs['S'] = lossS
    writer.add_scalars('main',scs, step)



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













