"""Build the separate 4060 fusion library without touching current binaries."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda-root',type=Path)
    p.add_argument('--tensorrt-include',type=Path)
    p.add_argument('--tensorrt-library',type=Path)
    p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];native=root/'native/libtorch'
    cache={};previous=native/'build/optimized/CMakeCache.txt'
    if previous.exists():
        for line in previous.read_text().splitlines():
            if not line.startswith(('#','//')) and ':' in line and '=' in line:
                cache[line.split(':',1)[0]]=line.split('=',1)[1]
    compiler=cache.get('CMAKE_CUDA_COMPILER') or shutil.which('nvcc')
    cuda=a.cuda_root or (Path(compiler).resolve().parent.parent if compiler else None)
    if cuda is None:raise ValueError('Provide --cuda-root for the installed CUDA toolkit')
    include=a.tensorrt_include or Path(cache.get('TENSORRT_INCLUDE',native/'build/tensorrt_include'))
    library=a.tensorrt_library or Path(cache.get('TENSORRT_LIBRARY',Path(sysconfig.get_paths()['purelib'])/'tensorrt_libs/libnvinfer.so.10'))
    for f in (cuda/'bin/nvcc',include/'NvInfer.h',library):
        if not f.is_file():raise FileNotFoundError(f)
    if a.jobs<1:raise ValueError('jobs must be positive')
    command=['cmake','-S',str(native),'-B',str(native/'build/fused'),'-DCMAKE_BUILD_TYPE=Release',
        '-DSTEREO_OPTIMIZED=ON','-DSTEREO_WITH_TENSORRT=ON','-DSTEREO_FUSIONS=ON',
        '-DCMAKE_CUDA_ARCHITECTURES=89',f'-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc',
        f'-DPython3_EXECUTABLE={sys.executable}',f'-DTENSORRT_INCLUDE={include}',
        f'-DTENSORRT_LIBRARY={library}',f'-DCUDA_RUNTIME_INCLUDE={cuda}/include']
    subprocess.run(command,check=True)
    subprocess.run(['cmake','--build',str(native/'build/fused'),'-j',str(a.jobs)],check=True)
if __name__=='__main__':main()
