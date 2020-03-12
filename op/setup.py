from setuptools import setup
import torch 
from torch.utils.cpp_extension import load, BuildExtension, CppExtension, CUDAExtension

import os 
import sys
import copy
"""
tokens = str(torch.__version__).split('.')
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))


modules = []

modules.extend([
            CUDAExtension('op', 
                [
	    'ext/op.cu', 'ext/op.cpp'
                    ], include_dirs=['ext/']
                )
        ])

setup(
        name='ops',
        ext_modules=modules,
        cmdclass={
            'build_ext': BuildExtension
            })
"""
op =load(name='op', sources=['ext/op.cpp',  'ext/op.cu'], verbose=True)
