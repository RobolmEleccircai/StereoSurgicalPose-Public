#pragma once
#include <ATen/ATen.h>
at::Tensor stereo_position_cuda(const at::Tensor& uv, const at::Tensor& z,
                               const at::Tensor& inverse, const at::Tensor& distortion);
