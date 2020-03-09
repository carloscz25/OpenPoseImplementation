from helpers import bodyparts, skeleton, skeletonnames, vectormodule
import numpy as np
from munkres import Munkres
import math


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
    S_READONLY = Snumpy.copy()
    Snumpy = Snumpy.flatten()
    for l in range(len(Snumpy)):
        val = Snumpy[l]
        if (val > confidencemappartsthreshold):
            i = math.floor(l / (S.shape[1]*S.shape[2]))
            j = 0


    #getting D
    for i in range(len(S_READONLY)):
        counter=0
        for j in (range(len(S_READONLY[i]))):
            for k in (range(len(S_READONLY[i,j]))):
                val = S_READONLY[i,j,k]
                if (val > confidencemappartsthreshold):
                    D[i,Dcounters[i]] = (j,k) #for each part i, we cound and store the position of the counter-th part found in the image
                    Dcounters[i] +=1
    #building E Matrix holding E-values for each
    #candidate limb
    associations = {}
    for i, limb in enumerate(skeleton):
        partindexfrom, partindexto = limb[0]-1, limb[1]-1 #limb part indexes start from 1
        D1, D2 = D[partindexfrom], D[partindexto]
        Dcounters1, Dcounters2 = Dcounters[partindexfrom], Dcounters[partindexto]
        if ((Dcounters1 == 0) | (Dcounters2 == 0)):#if any has no-points, makes no sense look at this limb
            continue
        EMtx = np.zeros((Dcounters1, Dcounters2), dtype=float)
        Lpart = L[i]
        #filling EMtx
        for a in range(Dcounters1):
            for b in range(Dcounters2):
                dj1 = D1[a]
                dj2 = D2[b]
                if ((dj1[0] - dj2[0])== 0) & ((dj1[1] - dj2[1])==0):
                    continue
                EMtx[a,b] = E(dj1, dj2, i, L)
        maxval = np.max(EMtx)
        #once filled we need to find the best combination of a's and b's
        # by finding the max values of E for a and b combinations with the
        # constraint that a_i can only be connected to b_j and the same
        # rules for b values
        # Munkres or Hungarian Algorithm
        munkres = Munkres()
        EMtx2 = munkres.make_cost_matrix(EMtx, lambda  profit: profit - maxval)
        indices = munkres.compute(EMtx2)

        #storing index association
        associations[i] = indices
    return associations





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


