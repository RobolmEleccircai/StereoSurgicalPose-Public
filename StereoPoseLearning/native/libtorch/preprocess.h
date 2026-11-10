#pragma once
#include <torch/torch.h>

// Current deployment: two RGB 1280x720 images -> 640x360 global input and
// unchanged 1280x720 local input. All pixel processing uses CUDA ATen ops.
class StereoPreprocess {
public:
    explicit StereoPreprocess(const torch::Tensor& normalization_lut, bool fused=false);
    torch::Tensor host_input() const { return host_; }
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run();
private:
    bool fused_;
    torch::Tensor horizontal_, small_, normalized_, xi_, xw_, yi_, yw_;
    torch::Tensor host_, raw_, lut_, channels_;
    std::vector<torch::Tensor> x_indices_, x_weights_, y_indices_, y_weights_;
};
