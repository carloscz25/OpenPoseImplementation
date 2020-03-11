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
counter = -1
for im, ann, S, L, impath in dataloader:
    counter += 1
    # print('Annotations', ann['annotations'])
    associations, D, Dcounters = performmultiparsing(S[0], L[0])
    img = getimagereconstructed_from_SLFields(im, associations, D)

    cv2.imshow('w', img)
    cv2.waitKey(0)
