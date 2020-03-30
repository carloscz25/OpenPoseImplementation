import scipy.io
import numpy as np
from helpers import *
import json

def turnimagenametoid(imagename):
    indr = imagename.rindex(".jpg", 0)
    im = imagename[0:indr]
    im = int(im)
    return im

imageanns = {'source':'MPiiDataset'}
imageanns['imagepath'] = '/home/carlos/PycharmProjects/PublicDatasets/MPII/images'
imageanns['imagenamelen'] = 9


annotationspath = '/home/carlos/PycharmProjects/PublicDatasets/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'


mat2 = scipy.io.loadmat(annotationspath, struct_as_record=False)


imagenumber = mat2['RELEASE'][0,0].annolist.shape[1]
for i in range(imagenumber):
    imagestruct = mat2['RELEASE'][0,0].annolist[0,i]
    imagename = imagestruct.image[0,0].name[0]
    dictanns = {}
    imageid = turnimagenametoid(imagename)
    imageanns[imageid] = dictanns
    dictanns['image_id'] = imageid
    # dictanns['image_name'] = imagename
    dictanns['annotations'] = []

    import os
    # im = cv2.imread(os.path.join(imageanns['imagepath'],imageurl(imageid, 'MPiiDataset')))
    # cv2.imshow('w', im)
    # cv2.waitKey(0)

    numberofpersons = mat2['RELEASE'][0,0].annolist[0,i].annorect.shape[1]
    for j in range(numberofpersons):
        persondict = {'id':j}
        dictanns['annotations'].append(persondict)
        pointsstruct = mat2['RELEASE'][0,0].annolist[0,i].annorect[0,j]
        if (indexof(pointsstruct._fieldnames, 'x1')!=-1):
            headrectangle = [pointsstruct.x1[0,0],pointsstruct.y1[0,0],pointsstruct.x2[0,0],pointsstruct.y2[0,0]]
        if (indexof(pointsstruct._fieldnames, 'annopoints') != -1):
            if (pointsstruct.annopoints.shape == (0,0))==True:
                continue
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
                    # is_visible = int(pstr.is_visible[0,0])
                    is_visible = int(pstr.is_visible[0])
                except:
                    is_visible = int(-1)
                if is_visible == True:
                    is_visible = 2
                elif is_visible == False:
                    is_visible = 1
                elif is_visible == -1:
                    is_visible = 0

                points[id] = (x,y,is_visible)
            lp = np.zeros((36), dtype=np.uint16)

            for k in points.keys():
                mapped_k = getpartindex(3,0,k)
                if mapped_k == -1:
                    continue
                # lp.append(points[k][0])
                lp[3 * mapped_k] = points[k][0]
                # lp.append(points[k][1])
                lp[(3*mapped_k)+1] = points[k][1]
                # lp.append(points[k][2])
                lp[(3 * mapped_k) + 2] = points[k][2]
                # cv2.circle(im, (points[k][0], points[k][1]), 2, (255,255,255), thickness=1)
                # cv2.putText(im, str(mapped_k), (points[k][0], points[k][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

            # cv2.imshow('w', im)
            # cv2.waitKey(0)
            persondict['keypoints'] = lp.tolist()
            persondict['bbox'] = bbx
            persondict['num_keypoints'] = int(num_keypoints)
    print('image n#' + str(i))
print('dumping...')
# json.dump(imageanns, open('/home/carlos/PycharmProjects/OpenPose/trainmpii.json','w'))
jsonstr = json.dumps(imageanns)
with open('/home/carlos/PycharmProjects/OpenPose/trainmpii.json', 'w') as f:
    f.write(jsonstr)
f.close()
print('done!')
y=2