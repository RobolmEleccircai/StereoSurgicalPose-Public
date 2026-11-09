#pragma once
#include <ATen/ATen.h>
#include <NvInfer.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// One context per serial inference stream. Optimized outputs are reusable and
// must be consumed before the next forward. Reference outputs remain owned.
class TrtVisual {
 public:
  explicit TrtVisual(const std::string& path);
  std::tuple<at::Tensor,at::Tensor,at::Tensor> forward(const at::Tensor& images);
 private:
#ifdef STEREO_OPTIMIZED
  std::vector<at::Tensor> output_buffers_;
#endif
  struct Logger: nvinfer1::ILogger {
    void log(Severity severity,const char* message) noexcept override;
  } logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};
