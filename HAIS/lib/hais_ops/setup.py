from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='HAIS_OP',
    ext_modules=[
        CUDAExtension('HAIS_OP', [
            'src/hais_ops_api.cpp',

            'src/hais_ops.cpp',
            'src/cuda.cu'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)