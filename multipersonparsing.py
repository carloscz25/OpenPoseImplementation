from helpers import bodyparts, skeleton, skeletonnames, vectormodule
import numpy as np
from munkres import Munkres


confidencemappartsthreshold = 0.7


def performmultiparsing(S,L):
    '''Pass in the
    S=> part confidencemaps
    L=>PAF maps'''
    #Let Dj be the set of  candidate parts for multiple people for the jth part
    #building the D set
    D = np.zeros((S.shape[0],100, 2)) #(a,100,2) D must contain locations for the ath part, 100=max number of ath parts and 2 is the 2D dimension
    #getting D
    for i in range(len(S)):
        counter=0
        for j in (range(len(S[i]))):
            for k in (range(len(S[i,j]))):
                val = S[i,j,k]
                if (val > confidencemappartsthreshold):
                    D[i,counter] = S[i, j] #for each part i, we cound and store the position of the counter-th part found in the image
                    counter +=1
    #building E Matrix holding E-values for each
    #candidate limb
    for i, limb in enumerate(skeleton):
        partindexfrom, partindexto = limb[0], limb[1]
        D1, D2 = D[partindexfrom], D[partindexto]
        EMtx = np.zeros(D1.shape[0], D2.shape[0], dtype=float)
        Lpart = L[i]
        #filling EMtx
        for a, dj1 in enumerate(D1):
            for b, dj2 in enumerate(D2):
                EMtx[a,b] = E(dj1, dj2, i, L)
        maxval = np.max(EMtx, 2)
        #once filled we need to find the best combination of a's and b's
        # by finding the max values of E for a and b combinations with the
        # constraint that a_i can only be connected to b_j and the same
        # rules for b values
        # Munkres or Hungarian Algorithm
        munkres = Munkres()
        EMtx2 = munkres.make_cost_matrix(EMtx, lambda  profit: profit - maxval)
        indices = munkres.compute(EMtx2)

        #verification time!
        #pending





def E(dj1,dj2,limbindex,L):
    '''Measure of association between part detections'''
    steps = 10
    step = 1/steps
    acc = 0
    for i in range(10):
        interpolationpos = i*step
        pu = ((1-interpolationpos)*dj1) +(interpolationpos*dj2)
        Lu = L[limbindex][pu[0]][pu[1]]
        vecdiff = (dj2[0]-dj1[0]),(dj2[1]-dj1[1])
        vecmod = vectormodule(vecdiff)
        unitvec = vecdiff/vecmod
        acc += np.dot(Lu, unitvec)
    return acc


