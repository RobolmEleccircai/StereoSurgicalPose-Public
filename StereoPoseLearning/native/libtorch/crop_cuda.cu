#include "fusion_cuda.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace {
__device__ float resize_map(const float* low,int y,int x,int h,int w,float cutoff,float divisor) {
  float fy=fmaxf((float(h)/720)*(y+.5f)-.5f,0.f);
  float fx=fmaxf((float(w)/1280)*(x+.5f)-.5f,0.f);
  int y0=int(fy),x0=int(fx),y1=min(y0+1,h-1),x1=min(x0+1,w-1);
  float dy=fy-y0,dx=fx-x0;
  float p=(1-dy)*((1-dx)*low[y0*w+x0]+dx*low[y0*w+x1])+
              dy*((1-dx)*low[y1*w+x0]+dx*low[y1*w+x1]);
  return fminf(1.f,fmaxf(0.f,(p-cutoff)/divisor));
}
__global__ void mask_kernel(const float* low,float* out,int h,int w,float cutoff,float divisor,int dilation) {
  int n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=2*720*1280)return;
  int x=n%1280,y=(n/1280)%720,b=n/(720*1280);float v=0.f;
  for(int yy=max(0,y-dilation);yy<=min(719,y+dilation);++yy)
    for(int xx=max(0,x-dilation);xx<=min(1279,x+dilation);++xx)
      v=fmaxf(v,resize_map(low+b*h*w,yy,xx,h,w,cutoff,divisor));
  out[n]=v;
}
__device__ float pixel(const uint8_t* rgb,const float* probability,int b,int c,int y,int x) {
  if(x<0 || x>=1280 || y<0 || y>=720)return 0.f;
  float p=probability[(b*720+y)*1280+x];
  if(c==3)return p;
  float m=c==0?.485f:(c==1?.456f:.406f),s=c==0?.229f:(c==1?.224f:.225f);
  float value=float(rgb[((b*3+c)*720+y)*1280+x]);
  return ((value/255.f-m)/s)*p;
}
__global__ void crop_kernel(const uint8_t* rgb,const float* probability,const float* grid,float* out,int gh,int gw) {
  int n=blockIdx.x*blockDim.x+threadIdx.x,area=gh*gw;
  if(n>=2*4*area)return;
  int i=n%area,c=(n/area)%4,b=n/(4*area);
  float gx=grid[(b*area+i)*2],gy=grid[(b*area+i)*2+1];
  float x=((gx+1.f)*1280-1.f)/2.f,y=((gy+1.f)*720-1.f)/2.f;
  if(!isfinite(x)||!isfinite(y)||x < -2.f||x>1281.f||y < -2.f||y>721.f){out[n]=0;return;}
  int x0=int(floorf(x)),y0=int(floorf(y)),x1=x0+1,y1=y0+1;
  float nw=(x1-x)*(y1-y),ne=(x-x0)*(y1-y),sw=(x1-x)*(y-y0),se=(x-x0)*(y-y0);
  float value=0.f;
  value+=pixel(rgb,probability,b,c,y0,x0)*nw;
  value+=pixel(rgb,probability,b,c,y0,x1)*ne;
  value+=pixel(rgb,probability,b,c,y1,x0)*sw;
  value+=pixel(rgb,probability,b,c,y1,x1)*se;
  out[n]=value;
}
}
at::Tensor stereo_mask_cuda(const at::Tensor& probability,double cutoff,int dilation) {
  TORCH_CHECK(probability.is_cuda() && probability.scalar_type()==at::kFloat && probability.dim()==4 &&
      probability.size(0)==2 && probability.size(1)==1 && probability.size(2)>0 && probability.size(3)>0,
      "Fused mask expects 2x1xHxW CUDA FP32");
  TORCH_CHECK(cutoff>=0 && cutoff<1 && dilation>=0 && dilation<=8,"Invalid mask parameters");
  c10::cuda::CUDAGuard guard(probability.device());auto low=probability.contiguous();
  auto out=at::empty({2,1,720,1280},low.options());
  mask_kernel<<<(2*720*1280+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(low.data_ptr<float>(),out.data_ptr<float>(),
      low.size(2),low.size(3),static_cast<float>(cutoff),static_cast<float>(1-cutoff),dilation);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
at::Tensor stereo_crop_cuda(const at::Tensor& hires,const at::Tensor& mask,const at::Tensor& grid,double cutoff,int dilation) {
  TORCH_CHECK(hires.is_cuda() && mask.device()==hires.device() && grid.device()==hires.device() &&
      hires.scalar_type()==at::kByte && hires.numel()==2*3*720*1280 && grid.scalar_type()==at::kFloat &&
      grid.dim()==4 && grid.size(0)==2 && grid.size(3)==2,"Invalid fused crop inputs");
  c10::cuda::CUDAGuard guard(hires.device());
  // Keep the original sigmoid kernel; fusion starts at upsampling, not a new approximation.
  auto probability=stereo_mask_cuda(mask.sigmoid(),cutoff,dilation);
  auto rgb=hires.reshape({2,3,720,1280}).contiguous(),g=grid.contiguous();
  int gh=g.size(1),gw=g.size(2);auto out=at::empty({2,4,gh,gw},g.options());
  crop_kernel<<<(2*4*gh*gw+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(rgb.data_ptr<uint8_t>(),probability.data_ptr<float>(),g.data_ptr<float>(),out.data_ptr<float>(),gh,gw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
