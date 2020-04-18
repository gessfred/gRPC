from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='mpitoaster',
      ext_modules=[cpp_extension.CppExtension('mpitoaster',
                                             ['mpitoaster.cc'],
                                             extra_compile_args=["-fopenmp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
