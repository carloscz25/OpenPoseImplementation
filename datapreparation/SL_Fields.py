import numpy as np
import math
from helpers import bodyparts, skeleton, skeletonnames, vectormodule, common_bodyparts, common_skeleton


varianceforconfidencemaps = 1
threshold_distance_to_limb = 1 #distance to limb imagainary line for a given point to be considered to belong to the limb



def scoreconfidence(pointevaluated, keypoint):
    vectordiff = (pointevaluated[0] - keypoint[0],pointevaluated[1] - keypoint[1])
    norm = math.sqrt((vectordiff[0]**2) + (vectordiff[1]**2))
    squared = norm**2
    squaredwithvariance = squared / varianceforconfidencemaps
    squaredwithvariance *= -1
    res = math.exp(squaredwithvariance)
    return res



def createconfidencemapsforpartdetection(imageshape, ann):
    #the map will have the same dimension as the image
    s = np.zeros((len(common_bodyparts),imageshape[0], imageshape[1]), dtype=np.float)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    for bodypartindex in range(len(common_bodyparts)):
        for personindex in range(len(ann['annotations'])):
            #if image is not annotated skip
            if ('keypoints' in ann['annotations'][personindex])==False:
                continue
            kp = ann['annotations'][personindex]['keypoints']
            # we create a confidencemap for each jth-part and person
            kpi = kp[bodypartindex*3:(bodypartindex*3)+3]
            if (kpi[2]==0): #if 1=not visible but annotated, if 2=visible and annotated
                #if this value is 0, there's no valid annotation
                continue
            # if (kpi[2]==1):
            #     #we skip it aswell as it is not visible
            #     continue
            if ((kpi[2] == 2)|(kpi[2] == 1)):
                disteval = 6
                #annotation and image/map coords order differs so we need to assign 1->0 and 0->1
                x0,y0, x1, y1 = kpi[0]-disteval, kpi[1]-disteval, kpi[0]+disteval, kpi[1]+disteval
                for y in range(y1-y0):
                    for x in range(x1-x0):
                        py, px = (y0 + y), (x0 + x)
                        if (py <0) | (py >= s.shape[1]):
                            continue
                        if (px <0) | (px >= s.shape[2]):
                            continue
                        confidence = scoreconfidence((px,py), (kpi[0], kpi[1]))
                        if confidence > s[bodypartindex,y0+y,x0+x]:
                            s[bodypartindex,y0+y,x0+x] = confidence
    return s

def createconfidencemapsforpartaffinityfields(imageshape, ann):
    limbsfound = []
    #the map will have the same dimension as the image
    L = np.zeros((len(common_skeleton)*2,imageshape[0], imageshape[1]), dtype=np.float)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    for limbindex in range(len(common_skeleton)):
        for personindex in range(len(ann['annotations'])):
            # if image is not annotated skip
            if ('keypoints' in ann['annotations'][personindex]) == False:
                continue
            kp = ann['annotations'][personindex]['keypoints']
            # we calculate the difference of the 2 keypoints
            # indicated for the linb pointed by bodypartindexes
            indexfrom = common_skeleton[limbindex][0]-1
            indexto = common_skeleton[limbindex][1]-1
            kpifrom = kp[indexfrom*3:(indexfrom*3)+3]
            kpito = kp[indexto * 3:(indexto * 3) + 3]
            if ((kpifrom[2]==0)|(kpito[2]==0)): #if 1=not visible but annotated, if 2=visible and annotated
                #if this value is 0, there's no valid annotation
                continue
            computeKeypointInPartAffinityMap(L,limbindex,kpifrom, kpito)
            limbsfound.append(limbindex)
    return L

def computeKeypointInPartAffinityMap(L, limbindex, kpifrom, kpito):
    # calculate unit vector
    limbname = skeletonnames[limbindex]
    limb = (kpito[1] - kpifrom[1], kpito[0] - kpifrom[0]) #swith the axes since coco is annotated x,y but opencv&numpy works y,x
    limblength = vectormodule(limb)
    #parche=> considerar eliminacion
    if (limblength==0.0):
        limblength = 0.001
    limbunitvector = [i / limblength for i in limb]

    # traverse the affected window
    disteval = 4
    evalbox = (
    min(kpifrom[0], kpito[0]) - disteval, min(kpifrom[1], kpito[1]) - disteval, max(kpifrom[0], kpito[0]) + disteval,
    max(kpifrom[1], kpito[1]) + disteval)
    x0, y0, x1, y1 = evalbox[0], evalbox[1], evalbox[2], evalbox[3]
    if x0 < 0:
        x0= 0
    if y0 < 0:
        y0 = 0
    if x1 > (L.shape[2]-1):
        x1 = L.shape[2] - 1
    if y1 > L.shape[1]-1:
        y1 = L.shape[1]-1
    for y in range(y1 - y0):
        for x in range(x1 - x0):
            # check if current point (x,y) is on limb
            # 2 conditions must assert true
            c1, c2 = False, False
            # 1.{0<=dotproduct(v,(p-limb[from])<=limblength
            px, py = x + x0, y + y0
            p = (px, py)
            dpc1 = np.dot(limbunitvector, ((p[0] - kpifrom[0], p[1] - kpifrom[1])))
            c1 = ((dpc1 >= 0) & (dpc1 <= limblength))
            # 2. {abs(dotproduct(vT,(p - limb[from)) <= threshold_distance_to_limb} xT = x transposed
            limbunitvectorperpendicular = (limbunitvector[1], -limbunitvector[0])
            p_distance_to_limb = abs(np.dot(limbunitvectorperpendicular, ((p[0] - kpifrom[0], p[1] - kpifrom[1]))))
            c2 = p_distance_to_limb <= threshold_distance_to_limb

            if (c1 == True & c2 == True):
                L[(limbindex*2)][y + y0][x + x0] = limbunitvector[0]
                L[(limbindex*2)+1][y + y0][x + x0] = limbunitvector[1]








