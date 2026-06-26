import re
import warnings
from setuptools import setup, Extension, find_packages
from distutils import sysconfig
import numpy as np
from dslflagger import __version__


if re.search('gcc', sysconfig.get_config_var('CC')) is None:
    args = []
else:
    args = ['-fopenmp']

try:
    from Cython.Distutils import build_ext

    HAVE_CYTHON = True

except ImportError as e:
    warnings.warn("Cython not installed.")
    from distutils.command import build_ext

    HAVE_CYTHON = False

def cython_file(filename):
    filename = filename + ('.pyx' if HAVE_CYTHON else '.c')
    return filename


# SumThreshold extension
st_ext = Extension(
            'dslflagger.rfi._sum_threshold', [ cython_file('dslflagger/rfi/_sum_threshold') ],
            include_dirs=[ np.get_include() ],
            extra_compile_args=args,
            extra_link_args=args)

# SIR operator extension
sir_ext = Extension(
            'dslflagger.rfi.sir_operator', [ cython_file('dslflagger/rfi/sir_operator') ],
            include_dirs=[ np.get_include() ],
            extra_compile_args=args,
            extra_link_args=args)


setup(
    name = 'dslflagger',
    version = __version__,

    packages = find_packages(),
    ext_modules = [ st_ext, sir_ext ],
    # scripts = ['scripts/dslflagger'],
    
    install_requires=[
        'numpy',
        'h5py',
        'caput',
        'matplotlib',
    ],
    extras_require={
        'cython': ['cython'],
        'mpi': ['mpi4py>=1.3'],
    },

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "DSL RFI flagging package.",
    license = "GPL v3.0",
    url = "https://github.com/DSL-SDP/dslflagger",
)
