#pragma once
#include <ATen/ATen.h>
#include <NvInfer.h>
#include <memory>
#include <string>
#include <tuple>

// One context per serial inference stream. Outputs are independently owned
// Torch tensors, so debug results do not alias a later call's engine buffers.
class TrtVisual {
 public:
  explicit TrtVisual(const std::string& path);
  std::tuple<at::Tensor,at::Tensor,at::Tensor> forward(const at::Tensor& images);
 private:
  struct Logger: nvinfer1::ILogger {
    void log(Severity severity,const char* message) noexcept override;
  } logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};
