from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='q_cpp',
      ext_modules=[cpp_extension.CppExtension('q_cpp', ['q.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

