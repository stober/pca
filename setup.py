'''
@author: stober
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]

ext_modules = [Extension("pca.fast",["src/fast_pca.pyx"], include_dirs=np_inc)]

setup(name='pca',
      version='0.1',
      description='Principle Component Analysis',
      author='Jeremy Stober',
      author_email='stober@gmail.com',
      package_dir={'pca':'src'},
      packages=['pca'],
      cmdclass = {'build_ext' : build_ext},
      ext_modules = ext_modules
      )
