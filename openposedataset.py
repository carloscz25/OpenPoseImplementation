from torch.utils.data.dataset import *
import os
import torch
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




        esvalido = False
        while (esvalido == False):
            image_id = next(iterator)
            if (image_id.isdigit()==False):
                continue
            ann = self.annotations[datasetname][image_id]
            try:
                for a in ann['annotations']:
                    b = a['bbox']
                esvalido = True
                image_url = os.path.join(imagepath, self.imageurl(datasetname, image_id))
                im = cv2.imread(image_url)
                if (im ==None):
                    esvalido = False
            except:
                    pass







        original_image_dim = im.shape

        annadjusted = adjustannotationpoints(copy.deepcopy(ann), im.shape, (28, 28))
        impreprocessed = self.performtransforms(Image.fromarray(im))

        # imwann = self.__imageannotated__(im, ann)
        # print(image_url)
        S, L = createconfidencemapsforpartdetection((28, 28), annadjusted), createconfidencemapsforpartaffinityfields((28, 28), annadjusted)

        return (impreprocessed, annadjusted, torch.from_numpy(S), torch.from_numpy(L), image_url, torch.from_numpy(np.asarray(original_image_dim)))







