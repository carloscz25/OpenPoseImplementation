from helpers import bodyparts, skeleton, skeletonnames, vectormodule
import numpy as np
from munkres import Munkres
import math
from datapreparation.SL_Fields_Cython import get_D_set_from_S_field

confidencemappartsthreshold = 0.7


def performmultiparsing(S,L):
    '''Pass in the
    S=> part confidencemaps
    L=>PAF maps'''
    #Let Dj be the set of  candidate parts for multiple people for the jth part
    #building the D set
    D = np.zeros((S.shape[0],100, 2)) #(a,100,2) D must contain locations for the ath part, 100=max number of ath parts and 2 is the 2D dimension
    Dcounters = np.zeros(S.shape[0], np.uint8) #stores the number of points found for each part
    #speeding things up
    Snumpy = S.clone().detach().numpy()
    #cython implementation
    D, Dcounters = get_D_set_from_S_field(Snumpy, confidencemappartsthreshold)
    #building E Matrix holding E-values for each
    #candidate limb
    associations = {}
    for i, limb in enumerate(skeleton):
        partindexfrom, partindexto = limb[0]-1, limb[1]-1 #limb part indexes start from 1
        D1, D2 = D[partindexfrom], D[partindexto]
        Dcounters1, Dcounters2 = Dcounters[partindexfrom], Dcounters[partindexto]
        if ((Dcounters1 == 0) | (Dcounters2 == 0)):#if any has no-points, makes no sense look at this limb
            continue
        if (Dcounters1 != Dcounters2):
            continue
        EMtx = np.zeros((int(Dcounters1), int(Dcounters2)), dtype=float)
        Lpart = L[i]
        #filling EMtx
        for a in range(int(Dcounters1)):
            for b in range(int(Dcounters2)):
                dj1 = D1[a]
                dj2 = D2[b]
                if ((dj1[0] - dj2[0])== 0) & ((dj1[1] - dj2[1])==0):
                    continue
                EMtx[a,b] = E(dj1, dj2, i, L)
        #negate values for maximization
        EMtx *= -1
        #once filled we need to find the best combination of a's and b's
        # by finding the max values of E for a and b combinations with the
        # constraint that a_i can only be connected to b_j and the same
        # rules for b values
        # Munkres or Hungarian Algorithm
        munkres = Munkres()
        indices = munkres.compute(EMtx)

        #storing index association
        associations[i] = indices
    #with this info we have enough info to rebuild the skeletons for all the people in the image
    return associations, D, Dcounters





def E(dj1,dj2,limbindex,L):
    '''Measure of association between part detections'''
    steps = 10
    step = 1/steps
    acc = 0
    for i in range(10):
        interpolationpos = i*step
        pu = ((1-interpolationpos)*dj1) +(interpolationpos*dj2)
        Lu = L[limbindex][int(pu[0])][int(pu[1])]
        vecdiff = (dj2[0]-dj1[0]),(dj2[1]-dj1[1])
        vecmod = vectormodule(vecdiff)
        unitvec = (vecdiff[0]/vecmod, vecdiff[1]/vecmod)
        acc += np.dot(Lu, unitvec)
    return acc


