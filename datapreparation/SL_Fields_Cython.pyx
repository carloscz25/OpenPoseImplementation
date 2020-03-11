import numpy

#cython version
def get_D_set_from_S_field(S_param, confidencemappartsthreshold):
    # cdef double[:,:,:] S = S_param
    S = S_param
    D = numpy.zeros((len(S),100,2))
    Dcounters = numpy.zeros((len(S)))
    for i in range(len(S)):

        for j in (range(len(S[i]))):
            for k in (range(len(S[i,j]))):
                val = S[i,j,k]
                if (val > confidencemappartsthreshold):
                    index = int(Dcounters[i])
                    D[i,index] = (j,k) #for each part i, we cound and store the position of the counter-th part found in the image
                    Dcounters[i] += 1
    return D, Dcounters