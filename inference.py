import torch
from torch import from_numpy
import numpy as np
from helpers import *
from torch.utils.data import DataLoader
from openposedataset import *
from model import OpenPoseModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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

paths = {}
paths['local'] = ['/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017','/home/carlos/PycharmProjects/PublicDatasets/MPII/images']
paths['cloud'] = ['/mnt/disks/sdb/datasets/coco/train2017','/mnt/disks/sdb/datasets/mpii/images']
dataset = OpenPoseDataset(['coco','mpii'], [0.9,0.1], paths['local'], ['traincoco.json', 'trainmpii.json'])
dataloader = torch.utils.data.DataLoader(dataset, 1, collate_fn=collatefn)

model = OpenPoseModel()
model.to(device)
sd = torch.load(open('checkpoints/current.chp', 'rb'))
model.load_state_dict(sd)
model.eval()



for step, ([impreprocessed, Starget, Ltarget, original_image_dim], [ann, image_url]) in enumerate(dataloader):
    Ltarget = Ltarget.float()
    Starget = Starget.float()
    impreprocessed = impreprocessed.to(device)
    F = model.F(impreprocessed)

    # CPM part
    F = model.cpm1(F)
    F = model.cpm1prlu(F)
    F = model.cpm2(F)
    F = model.cpm2prlu(F)

    # run L stages
    # stage1
    L = model.L1(F)
    Linput = torch.cat((F, L), 1)
    L = model.L2(Linput)
    Linput = torch.cat((F, L), 1)
    L = model.L3(Linput)
    Linput = torch.cat((F, L), 1)
    L = model.L4(Linput)
    Linput = torch.cat((F, L), 1)
    L = model.L5(Linput)
    # S stages
    # stage1
    Sinput = torch.cat((F, L), 1)
    S = model.S1(Sinput)
    Sinput = torch.cat((F, L, S), 1)
    S = model.S2(Sinput)
    Sinput = torch.cat((F, L), 1)
    S = model.S3(Sinput)
    Sinput = torch.cat((F, L, S), 1)
    S = model.S4(Sinput)

    index = 0
    original_image = cv2.imread(image_url[index])

    im1 = getimagewithpartheatmaps(original_image, S[index])


    # adjustannotationpoints(annadjusted, (28,28), original_image_dim.numpy()[0])
    # im = getimagewithdisplayedannotations(cv2.imread(image_url[0]), ann)
    #
    # im2 = drawSvsStarget(S, Starget)
    # orig_im = cv2.imread(image_url[0])

    # cv2.imshow('orig', orig_im)
    cv2.imshow('im1', im1)
    cv2.waitKey(0)




