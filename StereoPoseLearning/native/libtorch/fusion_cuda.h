#pragma once
#include <ATen/ATen.h>
void stereo_resize_normalize_cuda(const at::Tensor& raw, const at::Tensor& lut,
    const at::Tensor& xi,const at::Tensor& xw,const at::Tensor& yi,const at::Tensor& yw,
    at::Tensor& horizontal,at::Tensor& small,at::Tensor& normalized);
