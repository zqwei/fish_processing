from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

setup(
    name="l1tfsolvers",
    ext_modules=cythonize(
        [Extension("solvers.pdas",
                   ["solvers/pdas.pyx"],
                   include_dirs=[numpy.get_include()],
                   libraries=["lapacke", "lapack", "blas"],
                   extra_compile_args=['-std=c99']),
         Extension("solvers.ipm", ["solvers/ipm.pyx"],
                   include_dirs=[numpy.get_include()],
                   libraries=["blas", "lapack", "m"],
                   extra_compile_args=['-std=c99']), ]
    ),
    cmdclass={"build_ext": build_ext},
    packages=["solvers", ],
)
