from torch.utils.data.dataset import *
import os
import cv2
import PIL.Image as Image
from pycocotools.coco import COCO
import cv2
import numpy as np
import json
from datapreparation.SL_Fields import createconfidencemapsforpartdetection, createconfidencemapsforpartaffinityfields

class CocoPoseDataset(IterableDataset):

    imagepath = ''
    annotationpath = ''
    filenames = []
    annotations = []
    annotationiterator = None
    transforms = None
    coco = None

    def __init__(self, imagepath, annotationpath, transforms):
        super(CocoPoseDataset).__init__()
        self.transforms = transforms
        self.imagepath = imagepath
        self.annotationpath = annotationpath
        self.annotations = json.loads(open(self.annotationpath,'r').read())
        self.annotationiterator = iter(self.annotations)

    def imageurl(self, image_id):
        v = '0'*(12-len(str(image_id)))+str(image_id)+'.jpg'
        return v

    def __iter__(self):

        return self



    def __next__(self):
        '''

        :return:im-> original image / ann -> annotations / imwann -> image with specific annotations overlayed (bbox, keypoints, skeleton)
        '''
        image_id = next(self.annotationiterator)
        image_url = os.path.join(self.imagepath, self.imageurl(image_id))
        ann = self.annotations[image_id]
        im = cv2.imread(image_url)
        # imwann = self.__imageannotated__(im, ann)
        S, L = createconfidencemapsforpartdetection(im, ann), createconfidencemapsforpartaffinityfields(im, ann)
        return (im, ann, S, L)

    # def __imageannotated__(self, im, ann):
    #     imwann = np.copy(im)
    #     for index in range(len(self.bodyparts)):
    #         bpname = self.bodyparts[index]
    #         pp = ann['keypoints'][(index*3):((index*3)+3)]
    #         if pp[2]!=0:
    #             color = 125
    #             if pp[2]==1:
    #                 color = 70
    #             cv2.circle(imwann, (pp[0], pp[1]),2, color, 2)

        # bbox = ann['bbox']
        # x = int(bbox[0])
        # y = int(bbox[1])
        # w = x + int(bbox[2])
        # h = y + int(bbox[3])
        # cv2.rectangle(imwann, (x, y), (w, h), 16581375, 2)
        # return imwann



