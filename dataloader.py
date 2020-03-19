from torch.utils.data.dataset import *
import os
import cv2
import PIL.Image as Image
from pycocotools.coco import COCO
import cv2
import numpy as np
import json
from datapreparation.SL_Fields import createconfidencemapsforpartdetection, createconfidencemapsforpartaffinityfields
from torchvision.transforms import *
from PIL import Image
import copy
from helpers import *

class OpenPoseDataset(IterableDataset):



    def __init__(self, datasetnames, weights, imagepaths, annotationpaths):
        super(OpenPoseDataset).__init__()
        self.inputimagesize = 224
        self.datasetnames = datasetnames
        self.weights = weights
        self.imagepaths = imagepaths
        self.annotationpaths = annotationpaths
        self.annotations = {}
        self.annotationiterators = {}
        for i, path in enumerate(self.annotationpaths):
            self.annotations[self.datasetnames[i]] = json.loads(open(path,'r').read())
            self.annotationiterators[self.datasetnames[i]] = iter(self.annotations[self.datasetnames[i]])

    def imageurl(self, datasetname, image_id):
        if (datasetname == 'coco'):
            v = '0'*(12-len(str(image_id)))+str(image_id)+'.jpg'
        if (datasetname == 'mpii'):
            v = '0' * (9 - len(str(image_id))) + str(image_id) + '.jpg'
        return v



    def __iter__(self):
        return self


    def performtransforms(self, features):
        comp = Compose([
            CenterCrop(self.inputimagesize),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ]
        )
        res = comp(features)
        return res


    def __next__(self):
        '''

        :return:im-> original image / ann -> annotations / imwann -> image with specific annotations overlayed (bbox, keypoints, skeleton)
        '''
        from random import random

        rand = random()

        acc = 0
        datasetname = None
        imagepath = None
        iterator = None
        for i, k in enumerate(self.annotationiterators.keys()):
            acc += self.weights[i]
            if rand < acc:
                iterator = self.annotationiterators[k]
                datasetname = self.datasetnames[i]
                imagepath = self.imagepaths[i]
                break

        image_id = next(iterator)
        while (image_id.isdigit()==False):
            image_id = next(iterator)

        image_url = os.path.join(imagepath, self.imageurl(datasetname, image_id))
        ann = self.annotations[datasetname][image_id]
        im = cv2.imread(image_url)
        annadjusted = adjustannotationpoints(copy.deepcopy(ann), im.shape, (224,224))
        impreprocessed = self.performtransforms(Image.fromarray(im))

        # imwann = self.__imageannotated__(im, ann)
        print(image_url)
        S, L = createconfidencemapsforpartdetection((224,224), annadjusted), createconfidencemapsforpartaffinityfields((224,224), annadjusted)

        #return 0=original image, 1=preprocessed image, 2=image resized to input but unnormalized for visualization purposes
        #3 original annotations, 4=size adjusted annotations, 5 S size adjusted, 6 L size adjusted, 7 image url
        return [im, impreprocessed, cv2.resize(im, (self.inputimagesize, self.inputimagesize)), ann, annadjusted, S, L, image_url]







