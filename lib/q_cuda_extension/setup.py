from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='q_general_cpp',
      ext_modules=[cpp_extension.CppExtension('q_general_cpp',
                                             ['q_general.cpp'],
                                             extra_compile_args=["-fopenmp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
