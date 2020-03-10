from dataloader import CocoPoseDataset
from torch.utils.data import *
from helpers import *
import cv2
from multipersonparsing import performmultiparsing
from datapreparation.SL_Fields import createconfidencemapsforpartaffinityfields, createconfidencemapsforpartdetection

imagepath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017'
annotationpath = 'train.json'

dataset = CocoPoseDataset(imagepath, annotationpath, None)

dataloader = DataLoader(dataset, batch_size=1)

for im, ann, S, L in dataloader:
    # print('Annotations', ann['annotations'])
    associations, D, Dcounters = performmultiparsing(S[0], L[0])
    img = im.clone().detach().numpy()[0]
    for ii in skeleton:
