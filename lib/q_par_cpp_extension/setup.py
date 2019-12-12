from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='q_par_cpp',
      ext_modules=[cpp_extension.CppExtension('q_par_cpp', 
					     ['q_par.cpp'],
					     extra_compile_args=["-fopenmp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

