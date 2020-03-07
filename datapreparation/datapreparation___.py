import json
import os
import cv2
from helpers import *
import numpy as np

annotationpath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/annotations/person_keypoints_train2017.json'
imagepath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017'

bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]





preparing = False

if preparing:
    imageanns = {}

    with open(annotationpath, 'r') as f:
        jsonstr = f.read()
    f.close()

    anns = json.loads(jsonstr)

    for i in range(len(anns['annotations'])):
        imageann = imageanns.get(anns['annotations'][i]['image_id'])
        if imageann is None:
            imageann = {}
            imageann['image_id'] = anns['annotations'][i]['image_id']
            imageann['annotations'] = []
            imageanns[anns['annotations'][i]['image_id']] = imageann
        dict = {}
        imageann['annotations'].append(dict)
        dict['id'] = anns['annotations'][i]['id']
        dict['bbox'] = anns['annotations'][i]['bbox']
        dict['keypoints'] = anns['annotations'][i]['keypoints']
        dict['num_keypoints'] = anns['annotations'][i]['num_keypoints']

    jsonstr = json.dumps(imageanns)
    with open('train.json', 'w') as f:
        f.write(jsonstr)
    f.close()

    y = 2;
else:
    #reading the prepared data
    jsondata = open('train.json', 'r').read()
    imageanns = json.loads(jsondata)
    for k in imageanns.keys():
        path = os.path.join(imagepath, imageurl(k))
        img = cv2.imread(path)
        if img is not None:
            showimageandannotations(img, imageanns[k])
        else:
            print('image with id :' + str(k) +' is missing')
