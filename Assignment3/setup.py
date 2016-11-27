from distutils.core import setup
from Cython.Build import cythonize
import numpy

# Normally, one compiles cython extended code with .pyx ending
setup(
        ext_modules=cythonize("svmTrainDCGA2_cyt.pyx"),
        include_dirs=[numpy.get_include()]
        )
