from setuptools import setup, find_packages


from dslflagger import __version__

setup(
    name = 'dslflagger',
    version = __version__,

    packages = find_packages(),
    # scripts = ['scripts/dslflagger'],
    
    install_requires=[
        'numpy',
        'h5py',
        'caput',
        'matplotlib',
    ],

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "DSL RFI flagging package.",
    license = "GPL v3.0",
    url = "https://github.com/DSL/dslflagger",
)
