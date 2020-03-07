from dataloader import CocoPoseDataset
from torch.utils.data import *
from helpers import *
import cv2

imagepath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/val2017'
annotationpath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/annotations/person_keypoints_val2017.json'

dataset = CocoPoseDataset(imagepath, annotationpath, None)

dataloader = DataLoader(dataset, batch_size=1)

for i, o in enumerate(dataloader):
    (im, ann, imwann) = o
    imwann_ = imwann.clone().detach().numpy()
    cv2.imshow('w', imwann_[0])
    cv2.waitKey(1000)


y = 2