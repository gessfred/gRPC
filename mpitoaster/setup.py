from setuptools import setup
import torch
from torch.utils.cpp_extension import load, BuildExtension, CppExtension, CUDAExtension

setup(name='mpitoaster',
      ext_modules=[CUDAExtension('mpitoaster',
                                             ['mpitoaster.cc'],
                                             extra_compile_args=["-fopenmp", "-w"])],
      cmdclass={'build_ext': BuildExtension})