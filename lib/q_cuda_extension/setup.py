from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='q_cuda',
      ext_modules=[cpp_extension.CUDAExtension('q_cuda',
                                             ['q_cuda.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
