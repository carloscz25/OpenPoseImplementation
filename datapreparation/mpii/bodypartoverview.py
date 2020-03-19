


#Since I am having some confusion on the bodypart enumeration in the mpii dataset, let this file serve to check the exact index enumeration of the
#different bodyparts labelled in the MPii dataset

import scipy.io
import numpy as np
from helpers import *
import json

def turnimagenametoid(imagename):
    indr = imagename.rindex(".jpg", 0)
    im = imagename[0:indr]
    im = int(im)
    return im


imagepath = '/home/carlos/PycharmProjects/PublicDatasets/MPII/images'
imagenamelen = 9
annotationspath = '/home/carlos/PycharmProjects/PublicDatasets/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'

mat2 = scipy.io.loadmat(annotationspath, struct_as_record=False)

imagenumber = mat2['RELEASE'][0,0].annolist.shape[1]
for i in range(imagenumber):
    imagestruct = mat2['RELEASE'][0,0].annolist[0,i]
    imagename = imagestruct.image[0,0].name[0]
    dictanns = {}
    imageid = turnimagenametoid(imagename)

    import os
    imageuri = os.path.join(imagepath,imageurl(imageid, 'MPiiDataset'))
    im = cv2.imread(imageuri)
    cv2.putText(im, imageuri, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), thickness=1)

    numberofpersons = mat2['RELEASE'][0,0].annolist[0,i].annorect.shape[1]
    for j in range(numberofpersons):
        pointsstruct = mat2['RELEASE'][0,0].annolist[0,i].annorect[0,j]
        if (indexof(pointsstruct._fieldnames, 'x1')!=-1):
            headrectangle = [pointsstruct.x1[0,0],pointsstruct.y1[0,0],pointsstruct.x2[0,0],pointsstruct.y2[0,0]]
        if (indexof(pointsstruct._fieldnames, 'annopoints') != -1):
            points = {}
            bbx = [10000,10000,0,0]
            num_keypoints = pointsstruct.annopoints[0,0].point.shape[1]
            for p in range(num_keypoints):
                pstr = pointsstruct.annopoints[0,0].point[0,p]
                id = pstr.id[0,0]
                x = int(pstr.x[0,0])
                y = int(pstr.y[0, 0])
                bbx[0] = min(bbx[0], x)
                bbx[1] = min(bbx[1], y)
                bbx[2] = max(bbx[2], x)
                bbx[3] = max(bbx[3], y)
                try:
                    is_visible = int(pstr.is_visible[0,0])
                except:
                    is_visible = int(-1)
                if is_visible == True:
                    is_visible = 2
                elif is_visible == False:
                    is_visible = 1
                elif is_visible == -1:
                    is_visible = 0

                points[id] = (x,y,is_visible)

            for k in points.keys():
                cv2.circle(im, (points[k][0], points[k][1]), 2, (255,255,255), thickness=1)
                cv2.putText(im, str(k), (points[k][0], points[k][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
                inn = getpartindex(3,0,k)
                if inn !=-1:
                    cv2.putText(im, str(mappingtable[inn][1]), (points[k][0]+10, points[k][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

            cv2.imshow('w', im)
            cv2.waitKey(0)

    print('image n#' + str(i))
    if i ==30:
        break

