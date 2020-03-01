from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nccl',
    ext_modules=[
        CUDAExtension(name='nccl', sources=['collectives.cc'], include_dirs=['build/include', '/usr/include/mpi'], library_dirs=['build/lib'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })#, packages=[''], packages_data=['../build/lib/libnccl.so'])