import numpy as np
import cv2
import math

#17
bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
#19
skeletonnames = ['left_tibia','left_femur','right_tibia','right_femur','left_2_right_hip','left_shoulder_hip','right_shoulder_hip','left_right_shoulder','left_shoulder_elbow', 'right_shoulder_elbow','left_antebrazo', 'right_antebrazo','left_right_eye','nose_left_eye','nose_right_eye','left_eye_ear','right_eye_ear', 'left_ear_shoulder', 'right_ear_shoulder']


def imageurl(image_id):
    v = '0' * (12 - len(str(image_id))) + str(image_id) + '.jpg'
    return v

def vectormodule(vector):
    a = 0
    for i in vector:
        a += (i*i)
    res = math.sqrt(a)
    return res

def getimagewithdisplayedannotations(im, dict):
    im2 = np.copy(im)
    overlay0 = np.zeros(im.shape, dtype=np.uint8)
    overlay = np.zeros(im.shape, dtype=np.uint8)
    alpha = 1
    color = (0,255,0)
    im3 = im2.copy()
    anns = dict['annotations']
    for ann in anns:
        bbox = ann['bbox']
        x = int(bbox[0])
        y =  int(bbox[1])
        w = x + int(bbox[2])
        h = y + int(bbox[3])
        cv2.rectangle(overlay, (x, y), (w, h), color, 2)
        #keypoints

        for index in range(len(bodyparts)):
            bpname = bodyparts[index]
            pp = ann['keyoints'][(index*3):((index*3)+3)]
            if pp[2]!=0:
                color = (0,255,0)
                if pp[2]==1:
                    color = (0,0,255)
                cv2.circle(overlay, (pp[0], pp[1]),2, color, 2)
    cv2.addWeighted(im2, 1, overlay, 5, 0, dst=im3)
    return im3