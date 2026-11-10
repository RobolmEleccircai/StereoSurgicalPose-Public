#include "fusion_cuda.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
__global__ void horizontal_kernel(const uint8_t* raw,const int64_t* ix,const int32_t* weight,uint8_t* out) {
  int n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=2*720*640*3)return;
  int c=n%3,x=(n/3)%640,row=n/(3*640),sum=1<<21;
  #pragma unroll
  for(int t=0;t<4;++t)sum+=int(raw[(row*1280+ix[t*640+x])*3+c])*weight[t*640+x];
  out[n]=static_cast<uint8_t>(sum>>22);
}
__global__ void vertical_normalize_kernel(const uint8_t* horizontal,const int64_t* iy,
    const int32_t* weight,const float* lut,uint8_t* small,float* normalized) {
  int n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=2*360*640*3)return;
  int c=n%3,x=(n/3)%640,y=(n/(3*640))%360,b=n/(3*640*360),sum=1<<21;
  #pragma unroll
  for(int t=0;t<4;++t)sum+=int(horizontal[((b*720+iy[t*360+y])*640+x)*3+c])*weight[t*360+y];
  uint8_t v=static_cast<uint8_t>(sum>>22);
  small[n]=v;normalized[n]=lut[int(v)*3+c];
}
}
void stereo_resize_normalize_cuda(const at::Tensor& raw,const at::Tensor& lut,
    const at::Tensor& xi,const at::Tensor& xw,const at::Tensor& yi,const at::Tensor& yw,
    at::Tensor& horizontal,at::Tensor& small,at::Tensor& normalized) {
  TORCH_CHECK(raw.is_cuda() && raw.is_contiguous() && raw.scalar_type()==at::kByte &&
      raw.sizes()==at::IntArrayRef({2,720,1280,3}),"Fused resize requires contiguous stereo RGB uint8");
  c10::cuda::CUDAGuard guard(raw.device());
  auto stream=c10::cuda::getCurrentCUDAStream();
  horizontal_kernel<<<(2*720*640*3+255)/256,256,0,stream>>>(raw.data_ptr<uint8_t>(),xi.data_ptr<int64_t>(),xw.data_ptr<int32_t>(),horizontal.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  vertical_normalize_kernel<<<(2*360*640*3+255)/256,256,0,stream>>>(horizontal.data_ptr<uint8_t>(),yi.data_ptr<int64_t>(),yw.data_ptr<int32_t>(),lut.data_ptr<float>(),small.data_ptr<uint8_t>(),normalized.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
