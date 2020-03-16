import json
import os
import cv2
import numpy as np
from helpers import *

annotationpath = '/home/carlos/PycharmProjects/PublicDatasets/Coco/annotations/person_keypoints_train2017.json'

bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


imageanns = {'source':'CocoDataset'}
imageanns['imagepath'] = '/home/carlos/PycharmProjects/PublicDatasets/Coco/train2017'
imageanns['imagenamelen'] = 12

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
    #translating kepoints from coco values to jointmapping values
    kkpp = anns['annotations'][i]['keypoints']
    lp = np.zeros(36,np.uint16)
    for p in range(12):
        mapped_p = getpartindex(0,2,p)
        lp[3*p] = int(kkpp[3*mapped_p])
        lp[(3 * p)+1] = int(kkpp[(3 * mapped_p)+1])
        lp[(3 * p)+2] = int(kkpp[(3 * mapped_p)+2])


    dict['keypoints'] = lp.tolist()
    dict['num_keypoints'] = anns['annotations'][i]['num_keypoints']
    print('processing #' + str(i))
    if i==10:
        break
print('dumping...')
jsonstr = json.dumps(imageanns)
with open('../train.json', 'w') as f:
    f.write(jsonstr)
f.close()
print('done!')
