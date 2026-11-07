#include "model.h"
#include "preprocess.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(stereo_libtorch,m) {
  m.doc()="Eager LibTorch StereoPoseNet; no graph export, JIT or operator optimization";
  pybind11::class_<StereoPreprocess>(m,"StereoPreprocess")
      .def(pybind11::init<const torch::Tensor&>())
      .def("host_input",&StereoPreprocess::host_input)
      .def("run",&StereoPreprocess::run,pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<StereoNet>(m,"StereoNet")
      .def(pybind11::init<const TensorMap&,const NumericConfig&>())
      .def("forward",&StereoNet::forward,pybind11::arg("images"),pybind11::arg("K"),
           pybind11::arg("D"),pybind11::arg("hidden"),pybind11::arg("hires"),
           pybind11::arg("debug")=false,pybind11::call_guard<pybind11::gil_scoped_release>());
}
