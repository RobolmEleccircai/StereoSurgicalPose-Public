#pragma once
#include <ATen/ATen.h>
void stereo_resize_normalize_cuda(const at::Tensor& raw, const at::Tensor& lut,
    const at::Tensor& xi,const at::Tensor& xw,const at::Tensor& yi,const at::Tensor& yw,
    at::Tensor& horizontal,at::Tensor& small,at::Tensor& normalized);
at::Tensor stereo_mask_cuda(const at::Tensor& probability,double cutoff,int dilation);
at::Tensor stereo_crop_cuda(const at::Tensor& hires,const at::Tensor& mask,
    const at::Tensor& grid,double cutoff,int dilation);
at::Tensor stereo_crop_reference(const at::Tensor& hires,const at::Tensor& mask,
    const at::Tensor& grid,double cutoff,int dilation);
