#include "model.h"
#include "preprocess.h"
#ifdef STEREO_OPTIMIZED
#include "position_cuda.h"
#endif
#include "profile_ranges.h"
#ifdef STEREO_FUSIONS
#include "fusion_cuda.h"
#endif
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef STEREO_MODULE_NAME
#define STEREO_MODULE_NAME stereo_libtorch
#endif
PYBIND11_MODULE(STEREO_MODULE_NAME,m) {
#ifdef STEREO_WITH_TENSORRT
  m.doc()="LibTorch StereoPoseNet with optional C++ TensorRT visual frontend";
#else
  m.doc()="Eager LibTorch StereoPoseNet; no graph export, JIT or operator optimization";
#endif
#ifdef STEREO_PROFILE_RANGES
  m.def("set_stage_hook",[](pybind11::object callback) {
      if(callback.is_none()) { stereo_stage_hook={}; return; }
      stereo_stage_hook=[callback](const char* name) {
          pybind11::gil_scoped_acquire gil; callback(name);
      };
  });
#endif
#ifdef STEREO_FUSIONS
  m.def("crop_reference",&stereo_crop_reference);
  m.def("crop_cuda",&stereo_crop_cuda);
  m.def("mask_cuda",&stereo_mask_cuda);
#endif
  m.def("position_reference",&stereo_position_reference);
#ifdef STEREO_OPTIMIZED
  m.def("position_cuda",&stereo_position_cuda);
#endif
  pybind11::class_<StereoPreprocess>(m,"StereoPreprocess",pybind11::module_local())
      .def(pybind11::init<const torch::Tensor&,bool>(),pybind11::arg("normalization_lut"),pybind11::arg("fused")=false)
      .def("host_input",&StereoPreprocess::host_input)
      .def("run",&StereoPreprocess::run,pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<StereoNet>(m,"StereoNet",pybind11::module_local())
      .def(pybind11::init<const TensorMap&,const NumericConfig&>())
#ifdef STEREO_WITH_TENSORRT
      .def("load_visual_engine",&StereoNet::load_visual_engine)
#endif
      .def("set_calibration",&StereoNet::set_calibration)
      .def("forward",&StereoNet::forward,pybind11::arg("images"),pybind11::arg("K"),
           pybind11::arg("D"),pybind11::arg("hidden"),pybind11::arg("hires"),
           pybind11::arg("debug")=false,pybind11::call_guard<pybind11::gil_scoped_release>());
}
