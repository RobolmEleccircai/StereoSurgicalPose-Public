#include "trt_visual.h"
#include <c10/cuda/CUDAStream.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <vector>

void TrtVisual::Logger::log(Severity severity,const char* message) noexcept {
  if(severity<=Severity::kWARNING) std::cerr<<"[TensorRT] "<<message<<std::endl;
}
TrtVisual::TrtVisual(const std::string& path) {
  std::ifstream f(path,std::ios::binary);
  TORCH_CHECK(f.good(),"Cannot read TensorRT engine: ",path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)),{});
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  TORCH_CHECK(runtime_,"Cannot create TensorRT runtime");
  engine_.reset(runtime_->deserializeCudaEngine(bytes.data(),bytes.size()));
  TORCH_CHECK(engine_,"Cannot deserialize TensorRT engine");
  TORCH_CHECK(engine_->getNbIOTensors()==4,"Expected one input and three outputs");
  std::set<std::string> names;
  for(int i=0;i<engine_->getNbIOTensors();++i) names.insert(engine_->getIOTensorName(i));
  TORCH_CHECK(names==std::set<std::string>({"images","spatial","mask","heat"}),"Unexpected TensorRT interface");
  for(const auto& name:names) {
    TORCH_CHECK(engine_->getTensorDataType(name.c_str())==nvinfer1::DataType::kFLOAT,
                "TensorRT boundary tensors must be FP32");
    TORCH_CHECK(engine_->getTensorLocation(name.c_str())==nvinfer1::TensorLocation::kDEVICE,
                "TensorRT tensors must remain on GPU");
  }
  context_.reset(engine_->createExecutionContext());
  TORCH_CHECK(context_,"Cannot create TensorRT execution context");
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> TrtVisual::forward(const at::Tensor& images) {
  TORCH_CHECK(images.is_cuda() && images.get_device()==0 && images.is_contiguous() &&
              images.scalar_type()==at::kFloat,"Expected contiguous FP32 CUDA:0 input");
  auto dims=engine_->getTensorShape("images");
  TORCH_CHECK(dims.nbDims==images.dim(),"TensorRT input rank mismatch");
  for(int i=0;i<dims.nbDims;++i) TORCH_CHECK(dims.d[i]==images.size(i),"TensorRT input size mismatch");
  TORCH_CHECK(context_->setTensorAddress("images",images.data_ptr()),"Cannot bind TensorRT input");
#ifdef STEREO_OPTIMIZED
  auto& outputs=output_buffers_;
  if(outputs.empty()) {
#else
  std::vector<at::Tensor> outputs;
#endif
  for(const char* name:{"spatial","mask","heat"}) {
    auto shape=engine_->getTensorShape(name);
    std::vector<int64_t> sizes;
    for(int i=0;i<shape.nbDims;++i) {TORCH_CHECK(shape.d[i]>0,"Static TensorRT output required");sizes.push_back(shape.d[i]);}
    outputs.push_back(at::empty(sizes,images.options()));
    TORCH_CHECK(context_->setTensorAddress(name,outputs.back().data_ptr()),"Cannot bind TensorRT output");
  }
#ifdef STEREO_OPTIMIZED
  }
#endif
  TORCH_CHECK(outputs[0].sizes()==at::IntArrayRef({2,128,90,160}) &&
              outputs[1].sizes()==at::IntArrayRef({2,1,90,160}) &&
              outputs[2].sizes()==at::IntArrayRef({2,5,90,160}),"TensorRT output sizes differ from the trained model");
  auto stream=c10::cuda::getCurrentCUDAStream(0);
  TORCH_CHECK(context_->enqueueV3(stream.stream()),"TensorRT enqueue failed");
  return {outputs[0],outputs[1],outputs[2]};
}
