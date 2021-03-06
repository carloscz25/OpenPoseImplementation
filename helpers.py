import numpy as np
import cv2
import math

#COCO Information
#17
bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
#19
skeletonnames = ['left_tibia','left_femur','right_tibia','right_femur','left_2_right_hip','left_shoulder_hip','right_shoulder_hip','left_right_shoulder','left_shoulder_elbow', 'right_shoulder_elbow','left_antebrazo', 'right_antebrazo','left_right_eye','nose_left_eye','nose_right_eye','left_eye_ear','right_eye_ear', 'left_ear_shoulder', 'right_ear_shoulder']
skeletoncolors = []


#MPii Information
# id - joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
mpii_bodyparts = ['r ankle','r knee','r hip','l hip','l knee','l ankle','pelvis','thorax','upper neck','head top','r wrist','r elbow','r shoulder','l shoulder','l elbow','l wrist']



COCO = 2
MPii = 3

#mapping table
#cols:(index, name, cocoindex, mpiiindex)
mappingtable = []
mappingtable.append([0,'left ankle',15, 5])
mappingtable.append([1,'left knee',13, 4])
mappingtable.append([2,'left hip',11, 3])
mappingtable.append([3,'left shoulder',5, 13])
mappingtable.append([4,'left elbow',7, 14])
mappingtable.append([5,'left wrist',9, 15])
mappingtable.append([6,'right ankle',16, 0])
mappingtable.append([7,'right knee',14, 1])
mappingtable.append([8,'right hip',12, 2])
mappingtable.append([9,'right shoulder',6, 12])
mappingtable.append([10,'right elbow',8, 11])
mappingtable.append([11,'right wrist',10, 10])

partcolors = np.zeros((12,3), np.uint8)
partcolors[0] = (255,0,0)
partcolors[1] = (255,127,0)
partcolors[2] = (255,255,0)
partcolors[3] = (127,255,0)
partcolors[4] = (0,255,0)
partcolors[5] = (0,255,127)
partcolors[6] = (0,255,255)
partcolors[7] = (0,127,255)
partcolors[8] = (0,0,255)
partcolors[9] = (127,0,255)
partcolors[10] = (255,0,255)
partcolors[11] = (255,0,127)


# mappingtable.append([12,'head',0, 8]) #i don't like the correspondence between coco and mpii

common_bodyparts = [i[1] for i in mappingtable]
common_skeleton = [[1,2],[2,3],[3,4],[4,5],[5,6],[7,8],[8,9],[9,10],[10,11],[11,12]]
common_skeleton_names = ['left_tibia', 'left_femur', 'left_hip_shoulder', 'left_shoulder_elbow', 'left_elbow_wrist','right_tibia', 'right_femur', 'right_hip_shoulder', 'right_shoulder_elbow', 'right_elbow_wrist']

def getpartindex(fromdataset, todataset, partindex):
    '''
    Function to obtain part references from the different datasets employed
    :param fromdataset: 0-JointMapping / 2-CocoDataset / 3-MPiiDataset
    :param todataset: 0-JointMapping / 2-CocoDataset / 3-MPiiDataset
    :param partindex: [0-11]
    :return:
    '''
    for r in mappingtable:
        if r[fromdataset] == partindex:
            return r[todataset]
    return -1#unmapped


def imageurl(image_id, source):
    if source == 'CocoDataset':
        v = '0' * (12 - len(str(image_id))) + str(image_id) + '.jpg'
    if source == 'MPiiDataset':
        v = '0' * (9 - len(str(image_id))) + str(image_id) + '.jpg'
    return v

def vectormodule(vector):
    a = 0
    for i in vector:
        a += (i*i)
    res = math.sqrt(a)
    return res

def getimagewithpartheatmaps(im, S):
    '''im and S must have same width/height'''
    S3 = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)

    for i in range(len(S)):
        color = partcolors[i]
        for y in range(len(S[i])):
            for x in range(len(S[i][y])):
                yc = int(y*(im.shape[0]/S[i].shape[0]))
                xc = int(x * (im.shape[1] / S[i].shape[1]))
                if S[i,y,x] < 0.01:
                    continue
                S3[yc,xc,0] = int(color[0] * S[i,y,x])
                S3[yc, xc, 1] = int(color[1] * S[i, y, x])
                S3[yc, xc, 2] = int(color[2] * S[i, y, x])
                thresh = 220
                if ((S3[yc,xc,0]>=thresh) | (S3[yc,xc,1]>=thresh) | (S3[yc,xc,2]>=thresh)) :
                    pause = True
                    cv2.circle(S3, (xc,yc), 3, (int(color[0]),int(color[1]),int(color[2])), 2)


    im2 = np.copy(im)
    im3 = im.copy()
    cv2.addWeighted(im2, 1, S3, 5, 0, im3)
    return im3


def getimagewithdisplayedannotations(im, dict):
    im2 = np.copy(im)
    overlay0 = np.zeros(im.shape, dtype=np.uint8)
    overlay = np.zeros(im.shape, dtype=np.uint8)
    alpha = 1
    color = (0,255,0)
    im3 = im2.copy()
    anns = dict['annotations']
    for ann in anns:
        if ('bbox' in ann.keys())==False:
            continue
        bbox = ann['bbox']
        x = int(bbox[0])
        y =  int(bbox[1])
        w = x + int(bbox[2])
        h = y + int(bbox[3])
        # cv2.rectangle(overlay, (x, y), (w, h), color, 2)
        #keypoints
        index = 0
        for index in range(len(mappingtable)):
            bpname = mappingtable[index][1]
            pp = ann['keypoints'][(index*3):((index*3)+3)]
            if pp[2]!=0:
                color = (0,255,0)
                if pp[2]==1:
                    color = (0,0,255)
                cv2.circle(overlay, (pp[0], pp[1]),2, color, 2)
                cv2.putText(overlay, str(index), (pp[0],pp[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=2)
                cv2.putText(overlay, str(bpname), (pp[0]+10, pp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
    cv2.addWeighted(im2, 1, overlay, 5, 0, dst=im3)
    return im3

def getimagereconstructed_from_SLFields(im, associations, D):
    img = im.clone().detach().numpy()[0]
    for i, limb in enumerate(skeleton):
        partfrom, partto = limb[0] - 1, limb[1] - 1
        if associations.__contains__(i):
            for j in range(len(associations[i])):
                pointfrom, pointto = D[partfrom, associations[i][j][0]], D[partto, associations[i][j][1]]
                color = (255, 255, 0)
                cv2.line(img, (int(pointfrom[1]), int(pointfrom[0])), (int(pointto[1]), int(pointto[0])), color, 1)
    return img

def indexof(l, value):
    try:
        return l.index(value)
    except:
        return -1




def adjustannotationpoints(ann, fromsize, tosize):
    annotations = ann['annotations']
    for a in annotations:
        for d in a.keys():
            if (d=='bbox'):
                bbox = a['bbox']
                #en las anotaciones y y las imagenes las coordenadas x,y estan invertidas
                bbox[0] = int((tosize[1] / fromsize[1]) * bbox[0])
                bbox[1] = int((tosize[0] / fromsize[0]) * bbox[1])
                bbox[2] = int((tosize[1] / fromsize[1]) * bbox[2])
                bbox[3] = int((tosize[0] / fromsize[0]) * bbox[3])
            if (d=='keypoints'):
                l = a['keypoints']
                for i in range(0, len(l),3):
                    l[i] = int((tosize[1] / fromsize[1]) * l[i])
                    l[i+1] = int((tosize[0] / fromsize[0]) * l[i+1])
    return ann

def drawSvsStarget(S_batch, St_batch):
    cols = 4
    partsnumber = 12
    rows = int((12/cols) * 2)
    im = np.zeros((rows*28, cols*28), np.float)

    S = S_batch.clone().detach().numpy()
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            s_part_pic = S[i][j]
            mx, mn = np.max(s_part_pic), np.min(s_part_pic)
            for k in range(s_part_pic.shape[0]):
                for l in range(s_part_pic.shape[1]):
                    s_part_pic[k][l] = (s_part_pic[k][l]-mn)/(mx-mn)
            s_part_pic *= 255
        break
    S1 = S[0]
    St = St_batch.clone().detach().numpy()
    for i in range(St.shape[0]):
        for j in range(St.shape[1]):
            s_part_pic = St[i][j]
            mx, mn = np.max(s_part_pic), np.min(s_part_pic)
            for k in range(s_part_pic.shape[0]):
                for l in range(s_part_pic.shape[1]):
                    s_part_pic[k][l] = (s_part_pic[k][l] - mn) / (mx - mn)
            s_part_pic *= 255
        break
    S2 = St[0]


    currcol = 0
    currrow = 0
    for part in range(12):
        y1,y2 = currrow*28, (currrow+1)*28
        x1,x2 = currcol*28, currcol*28
        im[y1:(y1+28),x1:(x1+28)] = S1[part]
        im[y2:(y2+28), x2:(x2+28)] = S2[part]
        currcol +=1
        if currcol == 4:
            currcol, currrow = 0, currrow + 1

    return im








