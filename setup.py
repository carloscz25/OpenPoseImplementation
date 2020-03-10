from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("datapreparation/SL_Fields_Cython.pyx")
)