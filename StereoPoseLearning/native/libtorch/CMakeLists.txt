cmake_minimum_required(VERSION 3.18)
project(stereo_libtorch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import pathlib,torch; print(pathlib.Path(torch.__file__).parent); print(int(torch.compiled_with_cxx11_abi()))"
  OUTPUT_VARIABLE TORCH_INFO OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" TORCH_INFO "${TORCH_INFO}")
list(GET TORCH_INFO 0 TORCH_ROOT)
list(GET TORCH_INFO 1 TORCH_ABI)
Python3_add_library(stereo_libtorch MODULE model.cpp bindings.cpp preprocess.cpp)
target_include_directories(stereo_libtorch PRIVATE "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
target_compile_definitions(stereo_libtorch PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})
target_link_libraries(stereo_libtorch PRIVATE "${TORCH_ROOT}/lib/libtorch.so" "${TORCH_ROOT}/lib/libtorch_cpu.so"
  "${TORCH_ROOT}/lib/libtorch_cuda.so" "${TORCH_ROOT}/lib/libtorch_python.so" "${TORCH_ROOT}/lib/libc10.so")
set_target_properties(stereo_libtorch PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib" CXX_VISIBILITY_PRESET hidden)
