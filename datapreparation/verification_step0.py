import json
import os
import cv2
import numpy as np
from helpers import *

bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

imagepath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017'




#reading the prepared data
jsondata = open('../train.json', 'r').read()
imageanns = json.loads(jsondata)
for k in imageanns.keys():
    path = os.path.join(imagepath, imageurl(k))
    img = cv2.imread(path)
    if img is not None:
        showimageandannotations(img, imageanns[k])
    else:
        print('image with id :' + str(k) +' is missing')