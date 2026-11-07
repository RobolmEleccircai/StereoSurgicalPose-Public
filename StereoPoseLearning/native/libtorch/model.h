#pragma once
#include <ATen/ATen.h>
#include <torch/nn/modules/rnn.h>
#include <map>
#include <string>
#include <tuple>

// Eager LibTorch forward, matching StereoPoseNet. No serialized graph/runtime.
using TensorMap = std::map<std::string, at::Tensor>;
using NumericConfig = std::map<std::string, double>;
class StereoNet {
 public:
  StereoNet(const TensorMap& state, const NumericConfig& config);
  std::tuple<at::Tensor, at::Tensor, TensorMap> forward(
      const at::Tensor& images, const at::Tensor& K, const at::Tensor& D,
      const at::Tensor& hidden, const at::Tensor& hires, bool debug=false);
 private:
  TensorMap weights_;
  NumericConfig config_;
  std::map<std::string, torch::nn::GRU> recurrent_;
  const at::Tensor& w(const std::string& name) const;
  double cfg(const std::string& name, double fallback) const;
  at::Tensor conv(const at::Tensor&, const std::string&, int stride=1, int pad=0, int groups=1) const;
  at::Tensor linear(const at::Tensor&, const std::string&) const;
  at::Tensor ln(const at::Tensor&, const std::string&, double eps=1e-5) const;
  at::Tensor ln2(const at::Tensor&, const std::string&) const;
  at::Tensor gn(const at::Tensor&, const std::string&) const;
  at::Tensor mlp(const at::Tensor&, const std::string&) const;
  at::Tensor attention(const at::Tensor&, const at::Tensor&, const std::string&) const;
  std::tuple<at::Tensor,at::Tensor,at::Tensor> guided(
      const at::Tensor&,const at::Tensor&,const at::Tensor&) const;
  std::tuple<at::Tensor,at::Tensor,at::Tensor> local_branch(
      const at::Tensor&,const at::Tensor&,const at::Tensor&,const std::string&);
};
