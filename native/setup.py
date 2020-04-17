from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='native',
      ext_modules=[cpp_extension.CppExtension('native',
                                             ['native.cc'],
                                             extra_compile_args=["-fopenmp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
