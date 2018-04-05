from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

setup(
    name="pdas_sg2_C",
    ext_modules=cythonize([Extension("pdas_sg2_C.c_pdas_sg2",
                                     ["pdas_sg2_C/c_pdas_sg2.pyx"],
                                     include_dirs=[numpy.get_include()],
                                     libraries=["lapacke", "lapack", "blas"]), ]),
    cmdclass={"build_ext": build_ext},
    packages=["pdas_sg2_C", ],
)
