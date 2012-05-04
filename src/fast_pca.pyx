import numpy as np
import numpy.linalg as la
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

ITYPE = np.int
ctypedef np.int_t ITYPE_t

def compute_pca_fast(np.ndarray[DTYPE_t,ndim=2] data):
    """
    Modifies data input in place for efficiency. Returns m and s.
    """


    cdef int n = data.shape[0]
    cdef int isize = data.shape[1]
    cdef unsigned int i = 0
    cdef unsigned int j = 0

    cdef np.ndarray[DTYPE_t,ndim=1] m = np.mean(data, axis = 0)
    cdef np.ndarray[DTYPE_t,ndim=2] T = np.zeros((n,n))
    cdef np.ndarray[DTYPE_t,ndim=2] u = np.zeros((n,n))
    cdef np.ndarray[DTYPE_t,ndim=2] v = np.zeros((n,n))
    cdef np.ndarray[DTYPE_t,ndim=1] s = np.zeros(n)

    for i in xrange(n):
        data[i] = data[i] - m

    for i in xrange(n):
        for j in xrange(i+1):
            T[i,j] = np.dot(data[i],data[j])
            T[j,i] = T[i,j]

    u,s,v = np.linalg.svd(T)

    # Better for memory use if this is managed in place.

    for i in xrange(isize):
        data[:,i] = np.dot(data[:,i],u)

    for i in xrange(n):
        data[i,:] = data[i,:] / la.norm(data[i,:])

    return data,m,s,T,u
