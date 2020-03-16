import json
import os
import cv2
import numpy as np
from helpers import *




#reading the prepared data
jsondata = open('../../trainmpii.json', 'r').read()
imageanns = json.loads(jsondata)
imagepath = imageanns['imagepath']
source = imageanns['source']
for k in imageanns.keys():
    if k.isdigit()==False:
        continue
    path = os.path.join(imagepath, imageurl(k, source))
    img = cv2.imread(path)
    if img is not None:
        # showimageandannotations(img, imageanns[k])
        img2 = getimagewithdisplayedannotations(img, imageanns[k])
        cv2.imshow('w', img2)
        cv2.waitKey(0)
    else:
        print('image with id :' + str(k) +' is missing')