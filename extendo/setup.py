from setuptools import setup
import torch 
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys
import copy

os.environ["CC"] = "${CMAKE_C_COMPILER}"
os.environ["CXX"] = "${CMAKE_CXX_COMPILER}"
os.environ["CUDA_HOME"] = "${CUDA_TOOLKIT_ROOT_DIR}"

utility_dir = '${UTILITY_LIBRARY_DIRS}'
ops_dir = "${OPS_DIR}"

cuda_flags = '${CMAKE_CUDA_FLAGS}'.split(';')
print("cuda_flags = %s" % (' '.join(cuda_flags)))

include_dirs = [ops_dir]
lib_dirs = [utility_dir]
libs = ['utility'] 

tokens = str(torch.__version__).split('.')
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))

def add_prefix(filename):
    return os.path.join('${CMAKE_CURRENT_SOURCE_DIR}/src', filename)

modules = []

modules.extend([
    CppExtension('density_overflow_cpp', 
        [
            add_prefix('density_overflow.cpp')
            ], 
        include_dirs=copy.deepcopy(include_dirs), 
        library_dirs=copy.deepcopy(lib_dirs),
        libraries=copy.deepcopy(libs),
        extra_compile_args={
            'cxx' : [torch_major_version, torch_minor_version, '-fopenmp']
            })
    ])

if "${CUDA_FOUND}".upper() == 'TRUE': 
    modules.extend([
            CUDAExtension('density_overflow_cuda_thread_map', 
                [
                    add_prefix('density_overflow_cuda_thread_map.cpp'),
                    add_prefix('density_overflow_cuda_kernel.cu'), 
                    add_prefix('density_overflow_cuda_thread_map_kernel.cu')
                    ], 
                include_dirs=copy.deepcopy(include_dirs), 
                library_dirs=copy.deepcopy(lib_dirs),
                libraries=['cusparse', 'culibos'] + libs,
                extra_compile_args={
                    'cxx': ['-O2', torch_major_version, torch_minor_version], 
                    'nvcc': copy.deepcopy(cuda_flags)
                    }
                ),
            CUDAExtension('density_overflow_cuda_by_node', 
                [
                    add_prefix('density_overflow_cuda_by_node.cpp'),
                    add_prefix('density_overflow_cuda_by_node_kernel.cu')
                    ], 
                include_dirs=copy.deepcopy(include_dirs), 
                library_dirs=copy.deepcopy(lib_dirs),
                libraries=['cusparse', 'culibos'] + libs,
                extra_compile_args={
                    'cxx': ['-O2', torch_major_version, torch_minor_version], 
                    'nvcc': copy.deepcopy(cuda_flags)
                    }
                )
        ])

setup(
        name='density_overflow',
        ext_modules=modules,
        cmdclass={
            'build_ext': BuildExtension
            })