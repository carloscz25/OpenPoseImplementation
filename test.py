from dataloader import CocoPoseDataset
from torch.utils.data import *
from helpers import *
import cv2
from multipersonparsing import performmultiparsing
from datapreparation.SL_Fields import createconfidencemapsforpartaffinityfields, createconfidencemapsforpartdetection

imagepath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/val2017'
annotationpath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/annotations/person_keypoints_val2017.json'

dataset = CocoPoseDataset(imagepath, annotationpath, None)

dataloader = DataLoader(dataset, batch_size=1)

