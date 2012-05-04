#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: PCA.PY
Date: Tuesday, March 30 2010
Description: Principle component analysis.
"""

import numpy as np

# Note that PCA implementation in MDP cannot handle high dimensional
# data (at least the version I tested). Using a tricky set of
# convenient transformations, PCA can be applied to very high
# dimensional data as in this implementation.

def compute_pca(data):
    m = np.mean(data, axis=0)
    datac = np.array([obs - m for obs in data])
    T = np.dot(datac, datac.T)
    [u,s,v] = np.linalg.svd(T)

    # here iteration is over rows but the columns are the eigenvectors of T
    pcs = [np.dot(datac.T, item) for item in u.T ]

    # note that the eigenvectors are not normed after multiplication by T^T
    pcs = np.array([d / np.linalg.norm(d) for d in pcs])

    return pcs, m, s, T, u

def compute_projections(I,pcs,m):
    projections = []
    for i in I:
        w = []
        for p in pcs:
            w.append(np.dot(i - m, p))
        projections.append(w)
    return projections

def reconstruct(w, X, m,dim = 5):
    return np.dot(w[:dim],X[:dim,:]) + m

def normalize(samples, maxs = None):
    # Normalize data to [0,1] intervals. Supply the scale factor or
    # compute the maximum value among all the samples.

    if not maxs:
        maxs = np.max(samples)
    return np.array([np.ravel(s) / maxs for s in samples])
