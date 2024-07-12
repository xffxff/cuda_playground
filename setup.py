import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
]

# if os.environ.get("TORCH_CUDA_ARCH_LIST"):
#     # Let PyTorch builder to choose device to target for.
#     device_capability = ""
# else:
#     device_capability = torch.cuda.get_device_capability()
#     device_capability = f"{device_capability[0]}{device_capability[1]}"

# if device_capability:
#     nvcc_flags.extend([
#         f"--generate-code=arch=compute_{device_capability},code=sm_{device_capability}",
#         f"-DGROUPED_GEMM_DEVICE_CAPABILITY={device_capability}",
#     ])

nvcc_flags.extend([
    "--generate-code=arch=compute_80,code=sm_80",
])

ext_modules = [
    CUDAExtension(
        "cuda_playground_backend",
        ["csrc/layernorm.cu"],
        include_dirs = [
            f"{cwd}/third_party/cutlass/include/",
            f"{cwd}/csrc"
        ],
        extra_compile_args={
            "cxx": [
                "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
            ],
            "nvcc": nvcc_flags,
        }
    )
]

extra_deps = {}

extra_deps['dev'] = [
    'absl-py',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name="cuda_playground",
    version="0.0.1",
    author="zhou fan",
    description="My Cuda Playground",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    extras_require=extra_deps,
)