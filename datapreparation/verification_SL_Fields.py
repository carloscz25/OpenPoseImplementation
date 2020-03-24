from datapreparation.SL_Fields import createconfidencemapsforpartdetection, createconfidencemapsforpartaffinityfields
import json
import os
import cv2
from helpers import *

imagepath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017'




#load an image and its annotations from the prepared train.json/val.json
jsondata = open('../train.json', 'r').read()
imageanns = json.loads(jsondata)
for k in imageanns.keys():
    if k.isdigit()==False:
        continue
    path = os.path.join(imagepath, imageurl(k,'CocoDataset'))
    img = cv2.imread(path)
    anns = imageanns[k]
    iwda = getimagewithdisplayedannotations(img, anns)
    confidencemaps = createconfidencemapsforpartdetection(img.shape, anns)
    partaffinitymaps = createconfidencemapsforpartaffinityfields(img.shape, anns)
    #translate confidencemaps to grayscale
    aggregatedmap = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for bp in range(len(partaffinitymaps)):
        map = partaffinitymaps[bp]
        m = map.copy()
        m_ = np.zeros((m.shape[0], m.shape[1],3),dtype=np.uint8)
        hasdata = False
        for y in range(len(m)):
            for x in range(len(m[y])):
                if m[y][x] != 0.0:
                    m_[y][x] = (255,255,255)
                    hasdata = True
        if hasdata==False:
            continue

        aggregatedmap += m_
        # m = [255 for y in range(len(m)) for x in range(len(m[y])) ]
        # cv2.imshow('map_'+skeletonnames[bp], m)
    cv2.addWeighted(iwda, 1, aggregatedmap, 0.7, 0, dst=iwda)
    cv2.imshow('aggmap', iwda)
    cv2.waitKey(0)

    y=2



