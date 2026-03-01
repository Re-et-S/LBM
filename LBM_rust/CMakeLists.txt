cmake_minimum_required(VERSION 3.18)
project(LBM_CUDA LANGUAGES CXX CUDA)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "native")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Find necessary CUDA libraries (cuBLAS)
find_package(CUDAToolkit REQUIRED)

# Define source files for the library part (everything except main)
set(LBM_SOURCES
    src/kernels.cu
    src/transformer_ops.cu
    src/utils.cu
    src/lstm.cu
    src/checkpoint.cu
    src/generate.cu
    src/optimizer.cu
    src/loss.cu
)

# Create a static library containing the core logic
add_library(lbm_core STATIC ${LBM_SOURCES})
target_link_libraries(lbm_core PRIVATE CUDA::cublas)
set_target_properties(lbm_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Main Executable
add_executable(lbm_train src/main.cu)
target_link_libraries(lbm_train PRIVATE lbm_core CUDA::cublas)

# Tests Executable
add_executable(lbm_test src/test.cu)
target_link_libraries(lbm_test PRIVATE lbm_core CUDA::cublas)

# Enable testing
enable_testing()
add_test(NAME LBM_Tests COMMAND lbm_test)
