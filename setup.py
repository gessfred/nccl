from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm',
    ext_modules=[
        CUDAExtension(name='nccl', sources=['collectives.cc'],include_dirs=['../build/include'], library_dirs=['../build/lib'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })