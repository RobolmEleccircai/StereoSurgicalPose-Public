#include "position_cuda.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>

namespace {
__global__ void position_kernel(const float* uv, const float* z, const double* inv,
                                const double* ds, float* out, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const double* k=inv+9*i; const double* d=ds+5*i;
  double u=uv[2*i],v=uv[2*i+1];
  double w=k[6]*u+k[7]*v+k[8];
  double xd=(k[0]*u+k[1]*v+k[2])/w,yd=(k[3]*u+k[4]*v+k[5])/w;
  double x=xd,y=yd,k1=d[0],k2=d[1],p1=d[2],p2=d[3],k3=d[4];
  for(int j=0;j<10;++j) {
    double r2=x*x+y*y;
    double radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2;
    double tx=2*p1*x*y+p2*(r2+2*x*x),ty=p1*(r2+2*y*y)+2*p2*x*y;
    double g=k1+2*k2*r2+3*k3*r2*r2,rx=2*x*g,ry=2*y*g;
    double j11=radial+x*rx+2*p1*y+6*p2*x,j12=x*ry+2*p1*x+2*p2*y;
    double j21=y*rx+2*p1*x+2*p2*y,j22=radial+y*ry+6*p1*y+2*p2*x;
    double ex=x*radial+tx-xd,ey=y*radial+ty-yd,det=j11*j22-j12*j21;
    constexpr double eps=DBL_EPSILON*8;
    if(fabs(det)<eps)det=det>=0?eps:-eps;
    double sx=(j22*ex-j12*ey)/det,sy=(-j21*ex+j11*ey)/det;
    x=x-sx;y=y-sy;
  }
  out[3*i]=static_cast<float>(x*static_cast<double>(z[i]));
  out[3*i+1]=static_cast<float>(y*static_cast<double>(z[i]));
  out[3*i+2]=z[i];
}
}
at::Tensor stereo_position_cuda(const at::Tensor& uv,const at::Tensor& z,
                                const at::Tensor& inverse,const at::Tensor& distortion) {
  TORCH_CHECK(uv.is_cuda() && z.device()==uv.device() && inverse.device()==uv.device() &&
      distortion.device()==uv.device(),"Position tensors must share a CUDA device");
  TORCH_CHECK(uv.scalar_type()==at::kFloat && z.scalar_type()==at::kFloat &&
      inverse.scalar_type()==at::kDouble && distortion.scalar_type()==at::kDouble,
      "Position requires FP32 pixels/depth and FP64 calibration");
  int n=uv.numel()/2;
  TORCH_CHECK(uv.size(-1)==2 && z.numel()==n && inverse.numel()==9*n && distortion.numel()==5*n,
              "Invalid position dimensions");
  c10::cuda::CUDAGuard guard(uv.device());
  auto pixels=uv.contiguous(),depth=z.contiguous(),inv=inverse.contiguous(),d=distortion.contiguous();
  auto shape=uv.sizes().vec();shape.back()=3;
  auto out=at::empty(shape,uv.options());
  position_kernel<<<(n+31)/32,32,0,c10::cuda::getCurrentCUDAStream()>>>(pixels.data_ptr<float>(),
      depth.data_ptr<float>(),inv.data_ptr<double>(),d.data_ptr<double>(),out.data_ptr<float>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
