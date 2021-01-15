from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gconv_cuda',
    ext_modules=[
        CUDAExtension('gconv_cuda', [
            'gconv.cpp', 'gconv_sparse.cpp', 'gconv_kernel.cu',
            'gconv_backward.cu', 'gconv_sparse_kernel.cu',
            'gconv_sparse_backward.cu'
        ],
        extra_cuda_cflags=['-g','-G'])
    ],
    cmdclass={'build_ext': BuildExtension})
