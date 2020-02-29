from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm',
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'collectives.cc'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
