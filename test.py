from model import OpenPoseModel
from dataloader import CocoPoseDataset
from torch.utils.data import *

imagepath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017'
annotationpath = 'train.json'

dataset = CocoPoseDataset(imagepath, annotationpath, None)

dataloader = DataLoader(dataset, batch_size=1)

for t in dataloader:
    im = t[1]
    m = OpenPoseModel()

    m.forward(im)
    y = 2