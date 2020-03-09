from torch.utils.data.dataset import *
import os
import cv2
import PIL.Image as Image
from pycocotools.coco import COCO
import cv2
import numpy as np

class CocoPoseDataset(IterableDataset):

    imagepath = ''
    annotationpath = ''
    filenames = []
    annotations = []
    bodyparts = []
    skeleton = []
    annkeyiterator = None
    transforms = None
    coco = None

    def __init__(self, imagepath, annotationpath, transforms):
        super(CocoPoseDataset).__init__()
        self.transforms = transforms
        self.imagepath = imagepath
        self.annotationpath = annotationpath
        for dir, dirnames, filenames in os.walk(self.imagepath):
            self.filenames = filenames
        self.coco = COCO(self.annotationpath)
        self.annotations = self.coco.anns
        self.bodyparts = self.coco.cats[1]['keypoints']
        self.skeleton = self.coco.cats[1]['skeleton']
        self.annkeyiterator = self.annotations.keys().__iter__()


    def imageurl(self, ann):
        image_id = ann['image_id']
        v = '0'*(12-len(str(image_id)))+str(image_id)+'.jpg'
        return v

    def __iter__(self):

        return self



    def __next__(self):
        '''

        :return:im-> original image / ann -> annotations / imwann -> image with specific annotations overlayed (bbox, keypoints, skeleton)
        '''
        k = next(self.annkeyiterator)
        ann = self.annotations[k]
        image_url = os.path.join(self.imagepath, self.imageurl(ann))
        im = cv2.imread(image_url)
        imwann = self.__imageannotated__(im, ann)
        return (im, ann, imwann)

    def __imageannotated__(self, im, ann):
        imwann = np.copy(im)
        for index in range(len(self.bodyparts)):
            bpname = self.bodyparts[index]
            pp = ann['keypoints'][(index*3):((index*3)+3)]
            if pp[2]!=0:
                color = 125
                if pp[2]==1:
                    color = 70
                cv2.circle(imwann, (pp[0], pp[1]),2, color, 2)

        bbox = ann['bbox']
        x = int(bbox[0])
        y = int(bbox[1])
        w = x + int(bbox[2])
        h = y + int(bbox[3])
        cv2.rectangle(imwann, (x, y), (w, h), 16581375, 2)
        return imwann



