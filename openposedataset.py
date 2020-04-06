from torch.utils.data.dataset import *
import os
import torch
import cv2
import PIL.Image as Image

import cv2
import numpy as np
import json
from datapreparation.SL_Fields import createconfidencemapsforpartdetection, createconfidencemapsforpartaffinityfields
from torchvision.transforms import *
from PIL import Image
import copy
from helpers import *

class OpenPoseDataset(IterableDataset):



    def __init__(self, datasetnames, weights, imagepaths, annotationpaths, training_inference='training'):
        super(OpenPoseDataset).__init__()
        self.inputimagesize = 224
        self.datasetnames = datasetnames
        self.weights = weights
        self.imagepaths = imagepaths
        self.annotationpaths = annotationpaths
        self.annotations = {}
        self.training_inference = training_inference
        #initializing cursors
        self.datasetcursors = {}
        self.datasetlengths = {}
        self.datasetkeys = {}

        for i, path in enumerate(self.annotationpaths):
            annotations = json.loads(open(path,'r').read())
            self.annotations[self.datasetnames[i]] = annotations
            self.datasetlengths[self.datasetnames[i]] = len(annotations)
            self.datasetcursors[self.datasetnames[i]] = 0
            self.datasetkeys[self.datasetnames[i]] = list(annotations.keys())



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

        rand, acc = random(), 0
        for i, k in enumerate(self.datasetnames):
            acc += self.weights[i]
            if rand < acc:
                datasetname = self.datasetnames[i]
                imagepath = self.imagepaths[i]
                break
        esvalido = False
        while (esvalido == False):
            image_id = self.datasetkeys[datasetname][self.datasetcursors[datasetname]]
            self.datasetcursors[datasetname] += 1
            if self.datasetcursors[datasetname] == self.datasetlengths[datasetname]:
                self.datasetcursors[datasetname] = 0
            if (image_id.isdigit()==False):
                continue
            else:
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


        # acc = 0
        # datasetname = None
        # imagepath = None
        # iterator = None
        # for i, k in enumerate(self.annotationiterators.keys()):
        #     acc += self.weights[i]
        #     if rand < acc:
        #         iterator = self.annotationiterators[k]
        #         datasetname = self.datasetnames[i]
        #         imagepath = self.imagepaths[i]
        #         break
        #
        #
        # esvalido = False
        # while (esvalido == False):
        #     image_id = next(iterator)
        #     if (image_id.isdigit()==False):
        #         continue
        #     ann = self.annotations[datasetname][image_id]
        #     try:
        #         for a in ann['annotations']:
        #             b = a['bbox']
        #         esvalido = True
        #         image_url = os.path.join(imagepath, self.imageurl(datasetname, image_id))
        #         im = cv2.imread(image_url)
        #         if (im ==None):
        #             esvalido = False
        #     except:
        #             pass

        original_image_dim = im.shape

        annadjusted = adjustannotationpoints(copy.deepcopy(ann), im.shape, (28, 28))
        impreprocessed = self.performtransforms(Image.fromarray(im))

        # imwann = self.__imageannotated__(im, ann)
        # print(image_url)
        self.getnewiteration = False
        S, L = createconfidencemapsforpartdetection((28, 28), annadjusted), createconfidencemapsforpartaffinityfields((28, 28), annadjusted)
        if self.training_inference == 'training':
            return (impreprocessed, annadjusted, torch.from_numpy(S), torch.from_numpy(L), image_url, torch.from_numpy(np.asarray(original_image_dim)))
        if self.training_inference == 'inference':
            return (impreprocessed, annadjusted, ann, torch.from_numpy(S), torch.from_numpy(L), image_url, torch.from_numpy(np.asarray(original_image_dim)))








