"""Build the CUDA extension for TurboQuant acceleration."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='cuda_turboquant',
    ext_modules=[
        CUDAExtension(
            'cuda_turboquant',
            ['turboquant_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-std=c++17', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
