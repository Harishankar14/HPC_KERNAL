#include "detect/hardware_type.hpp"  //ehh could have used eextern, but anyways.

#ifdef USE_CUDA
#include <cuda_runtime.h>
bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}
#else
bool is_cuda_available() {
    return false;
}
#endif

#include <fstream>
#include <string>

bool supportsAVX_Linux() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("avx") != std::string::npos)
            return true;
    }
    return false;
}

HardwareType detectHardwareAutomatically() {
    if (is_cuda_available()) {
        return HardwareType::GPU;
    }

    if (supportsAVX_Linux()) {
        return HardwareType::SIMD;
    }

    return HardwareType::CPU;
}
