#include "fusion_cuda.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
// One CTA per landmark. Two passes retain normalize-before-coordinate-weighting.
// Recomputing scalar probabilities avoids materializing full probability/weight maps.
__global__ void landmark_kernel(const float* heat,float* xy,float* confidence,int h,int w) {
  __shared__ float sums[256],maxima[256],xs[256],ys[256];
  int lane=threadIdx.x,row=blockIdx.x,n=h*w;float den=0.f,peak=0.f;
  for(int i=lane;i<n;i+=256) {
    float p=1.f/(1.f+expf(-heat[row*n+i]));
    den+=powf(p,4.f)+1e-12f;peak=fmaxf(peak,p);
  }
  sums[lane]=den;maxima[lane]=peak;__syncthreads();
  for(int step=128;step;step/=2) {
    if(lane<step){sums[lane]+=sums[lane+step];maxima[lane]=fmaxf(maxima[lane],maxima[lane+step]);}
    __syncthreads();
  }
  den=fmaxf(sums[0],1e-12f);float x=0.f,y=0.f;
  for(int i=lane;i<n;i+=256) {
    float p=1.f/(1.f+expf(-heat[row*n+i]));
    float weight=(powf(p,4.f)+1e-12f)/den;
    x+=weight*((float(i%w)+.5f)*2.f/float(w)-1.f);
    y+=weight*((float(i/w)+.5f)*2.f/float(h)-1.f);
  }
  xs[lane]=x;ys[lane]=y;__syncthreads();
  for(int step=128;step;step/=2) {
    if(lane<step){xs[lane]+=xs[lane+step];ys[lane]+=ys[lane+step];}
    __syncthreads();
  }
  if(lane==0){xy[2*row]=xs[0];xy[2*row+1]=ys[0];confidence[row]=maxima[0];}
}
}
std::tuple<at::Tensor,at::Tensor> stereo_landmark_cuda(const at::Tensor& heat,double power) {
  TORCH_CHECK(heat.is_cuda() && heat.scalar_type()==at::kFloat && heat.sizes()==at::IntArrayRef({2,5,90,160}) && power==4,
      "Fused landmark reduction supports 2x5x90x160 FP32 with power 4");
  c10::cuda::CUDAGuard guard(heat.device());auto x=heat.contiguous();
  auto xy=at::empty({2,5,2},x.options()),confidence=at::empty({2,5},x.options());
  landmark_kernel<<<10,256,0,c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),xy.data_ptr<float>(),confidence.data_ptr<float>(),90,160);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {xy,confidence};
}
