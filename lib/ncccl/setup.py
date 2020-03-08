from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nccl',
    ext_modules=[
        CUDAExtension(name='nccl', sources=['ncccl.cc'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })#, packages=[''], packages_data=['../build/lib/libnccl.so'])