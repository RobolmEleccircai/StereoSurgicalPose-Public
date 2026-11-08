#include "preprocess.h"
#include "profile_ranges.h"
#include <cmath>

namespace {
// Pillow 11.3 RGB BILINEAR downsampling by two: separable four-tap filter,
// renormalized borders, 22-bit coefficients and uint8 rounding after EACH pass.
// See https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c
void axis(int size, int dimension, std::vector<torch::Tensor>& indices,
          std::vector<torch::Tensor>& weights) {
    for (int tap=0; tap<4; ++tap) {
        std::vector<int64_t> ix(size/2);
        std::vector<int32_t> co(size/2);
        for (int out=0; out<size/2; ++out) {
            double w;
            if (out==0) {
                ix[out]=std::min(tap,2);
                w=tap<2 ? 3.0/7 : (tap==2 ? 1.0/7 : 0);
            } else if (out==size/2-1) {
                ix[out]=size-3+std::min(tap,2);
                w=tap==0 ? 1.0/7 : (tap<3 ? 3.0/7 : 0);
            } else {
                ix[out]=2*out-1+tap;
                w=(tap==0 || tap==3) ? 1.0/8 : 3.0/8;
            }
            co[out]=static_cast<int32_t>(std::floor(w*(1<<22)+0.5));
        }
        indices.push_back(torch::tensor(ix,torch::kInt64).to(torch::kCUDA));
        std::vector<int64_t> shape(4,1); shape[dimension]=size/2;
        weights.push_back(torch::tensor(co,torch::kInt32).to(torch::kCUDA).view(shape));
    }
}

torch::Tensor pass(const torch::Tensor& input, int dimension,
                   const std::vector<torch::Tensor>& indices,
                   const std::vector<torch::Tensor>& weights) {
    torch::Tensor sum;
    for (int i=0; i<4; ++i) {
        auto value=input.index_select(dimension,indices[i]).to(torch::kInt32)*weights[i];
        if (i==0) sum=value.add_(1<<21); else sum.add_(value);
    }
    return torch::bitwise_right_shift(sum,22).to(torch::kUInt8);
}
}

StereoPreprocess::StereoPreprocess(const torch::Tensor& normalization_lut) {
    TORCH_CHECK(normalization_lut.sizes()==torch::IntArrayRef({256,3}) &&
                normalization_lut.scalar_type()==torch::kFloat32,"Expected 256x3 FP32 RGB normalization table");
    host_=torch::empty({2,720,1280,3},torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
    raw_=torch::empty({2,720,1280,3},torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    lut_=normalization_lut.contiguous().view({-1}).to(torch::kCUDA);
    channels_=torch::arange(3,torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)).view({1,1,1,3});
    axis(1280,2,x_indices_,x_weights_); axis(720,1,y_indices_,y_weights_);
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> StereoPreprocess::run() {
    torch::InferenceMode guard;
    // Caller owns the single in-flight frame and must not overwrite host_input
    // until its preceding CUDA work has completed (also on exceptional paths).
    ProfileStages profile;
    profile.next("stage::upload_rgb");
    raw_.copy_(host_,true);
    profile.next("stage::resize_horizontal");
    auto horizontal=pass(raw_,2,x_indices_,x_weights_);
    profile.next("stage::resize_vertical");
    auto small=pass(horizontal,1,y_indices_,y_weights_);
    horizontal=torch::Tensor();
    // Lookup on GPU preserves NumPy FP32 divide/subtract/divide rounding for
    // every possible uint8 RGB value, without uploading normalized images.
    profile.next("stage::normalize_layout");
    auto lookup=small.to(torch::kInt64).mul_(3).add_(channels_);
    auto normalized=lut_.index_select(0,lookup.reshape({-1})).view({2,360,640,3});
    auto images=normalized.permute({0,3,1,2}).unsqueeze(0).unsqueeze(0);
    auto hires=raw_.permute({0,3,1,2}).contiguous().unsqueeze(0).unsqueeze(0);
    return {images,hires,small};
}
