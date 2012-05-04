#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST.PY
Date: Thursday, May  3 2012
Description: Test PCA.
"""

import numpy
import pca.fast
import pca

x = numpy.random.randn(10,100)
y = x.copy()
pcs1, m1, s1,T1,u1 = pca.fast.compute_pca_fast(x)
pcs2, m2, s2,T2,u2 = pca.compute_pca(y)

print pcs2.shape
print x.shape
#print x - pcs
print numpy.allclose(m1,m2)
print numpy.allclose(s1,s2)
print numpy.allclose(T1,T2)
print numpy.allclose(x,y)
#print numpy.allclose(x,pcs2)
print numpy.allclose(pcs1[8],pcs2[8]) # last sig. pcs
print numpy.allclose(u1,u2)
#print pcs1 - pcs2
