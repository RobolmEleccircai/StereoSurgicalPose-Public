"""Build the optional optimized module without replacing reference libraries."""
import argparse
from pathlib import Path
import subprocess
import sys
import sysconfig

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-root',type=Path,default=Path('/usr/local/cuda'))
    parser.add_argument('--trt-include',type=Path)
    parser.add_argument('--trt-library',type=Path)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    import torch
    root=Path(__file__).resolve().parents[1]
    headers=args.trt_include or root/'native/libtorch/build/tensorrt_include'
    library=args.trt_library or Path(sysconfig.get_paths()['purelib'])/'tensorrt_libs/libnvinfer.so.10'
    for path in (headers/'NvInfer.h',library,args.cuda_root/'bin/nvcc'):
        if not path.exists():raise FileNotFoundError(path)
    capability=torch.cuda.get_device_capability()
    build=root/'native/libtorch/build/optimized'
    subprocess.run(['cmake','-S',str(root/'native/libtorch'),'-B',str(build),
        '-DSTEREO_OPTIMIZED=ON','-DSTEREO_WITH_TENSORRT=ON','-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CUDA_COMPILER='+str(args.cuda_root/'bin/nvcc'),
        '-DCMAKE_CUDA_ARCHITECTURES='+''.join(map(str,capability)),
        '-DPython3_EXECUTABLE='+sys.executable,'-DTENSORRT_INCLUDE='+str(headers),
        '-DTENSORRT_LIBRARY='+str(library),'-DCUDA_RUNTIME_INCLUDE='+str(args.cuda_root/'include')],check=True)
    subprocess.run(['cmake','--build',str(build),'-j',str(args.jobs)],check=True)
if __name__=='__main__':main()
