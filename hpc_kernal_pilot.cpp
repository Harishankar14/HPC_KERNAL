#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include "json.hpp" // nlohmann::json

using json = nlohmann::json;

// ------------------------- Hardware Detection -------------------------
enum class HardwareType { CPU, GPU, SIMD, ALL };

HardwareType detectHardwareFromConfig() {
    std::ifstream file("config.json");
    if (!file.is_open()) return HardwareType::CPU;
    json j;
    file >> j;
    std::string hw = j.value("kernel", "cpu");
    if (hw == "gpu") return HardwareType::GPU;
    if (hw == "simd") return HardwareType::SIMD;
    if (hw == "all") return HardwareType::ALL;
    return HardwareType::CPU;
}

// ------------------------- Thread Config ------------------------------
void configureThreadsFromConfig() {
    std::ifstream file("config.json");
    if (!file.is_open()) return;
    json j;
    file >> j;
    int threads = j.value("threads", -1);
    if (threads > 0) {
        omp_set_num_threads(threads);
        std::cout << "Using " << threads << " threads (OpenMP)\n";
    }
}

// ------------------------- Matrix Size Config -------------------------
size_t getMatrixSizeFromConfig() {
    std::ifstream file("config.json");
    if (!file.is_open()) return 256;
    json j;
    file >> j;
    return j.value("matrix_size", 256);
}

// ------------------------- Kernel Interface ---------------------------
class IKernel {
public:
    virtual void execute(const std::vector<std::vector<int>>&, const std::vector<std::vector<int>>&, std::vector<std::vector<int>>&) = 0;
    virtual const char* name() const = 0;
    virtual const char* version() const = 0;
    virtual ~IKernel() = default;
};

// ------------------------- CPU Kernel ---------------------------------
class CpuMatrixMul : public IKernel {
public:
    void execute(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) override {
        size_t N = A.size();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                for (size_t k = 0; k < N; ++k)
                    C[i][j] += A[i][k] * B[k][j];
    }
    const char* name() const override { return "CPU Matrix Mul"; }
    const char* version() const override { return "1.0"; }
};

// ------------------------- GPU Kernel (Simulated) ---------------------
class GpuMatrixMul : public IKernel {
public:
    void execute(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) override {
        size_t N = A.size();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                for (size_t k = 0; k < N; ++k)
                    C[i][j] += A[i][k] * B[k][j];
    }
    const char* name() const override { return "GPU Matrix Mul (Simulated)"; }
    const char* version() const override { return "1.1"; }
};

// ------------------------- SIMD (OpenMP) Kernel -----------------------
class SimdMatrixMul : public IKernel {
public:
    void execute(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) override {
        size_t N = A.size();
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(N); ++i)
            for (size_t j = 0; j < N; ++j)
                for (size_t k = 0; k < N; ++k)
                    C[i][j] += A[i][k] * B[k][j];
    }
    const char* name() const override { return "SIMD (OpenMP) Matrix Mul"; }
    const char* version() const override { return "2.0"; }
};

// ------------------------- Kernel Registry ----------------------------
class KernelRegistry {
public:
    static std::unique_ptr<IKernel> getKernel(HardwareType type) {
        switch (type) {
            case HardwareType::GPU: return std::make_unique<GpuMatrixMul>();
            case HardwareType::SIMD: return std::make_unique<SimdMatrixMul>();
            case HardwareType::CPU:
            default: return std::make_unique<CpuMatrixMul>();
        }
    }
};

// ------------------------- Utilities ----------------------------------
std::vector<std::vector<int>> generateMatrix(size_t N) {
    return std::vector<std::vector<int>>(N, std::vector<int>(N, 1));
}

// ------------------------- Benchmark ----------------------------------
void benchmarkAllKernels(size_t N, const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    for (HardwareType hwType : {HardwareType::CPU, HardwareType::GPU, HardwareType::SIMD}) {
        auto kernel = KernelRegistry::getKernel(hwType);
        std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

        auto start = std::chrono::high_resolution_clock::now();
        kernel->execute(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << kernel->name() << " | Time: " << elapsed << " ms | C[0][0] = " << C[0][0] << "\n";
    }
}

// ------------------------- Main ---------------------------------------
int main(int argc, char* argv[]) {
    std::string kernelArg = argc > 1 ? argv[1] : "";
    size_t N = argc > 2 ? std::stoi(argv[2]) : getMatrixSizeFromConfig();

    auto A = generateMatrix(N);
    auto B = generateMatrix(N);
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    HardwareType hw;
    if (kernelArg == "gpu") hw = HardwareType::GPU;
    else if (kernelArg == "simd") hw = HardwareType::SIMD;
    else if (kernelArg == "all") hw = HardwareType::ALL;
    else hw = detectHardwareFromConfig();

    configureThreadsFromConfig();

    if (hw == HardwareType::ALL) {
        benchmarkAllKernels(N, A, B);
    } else {
        auto kernel = KernelRegistry::getKernel(hw);
        std::cout << "Using Kernel: " << kernel->name() << " (v" << kernel->version() << ")\n";

        auto start = std::chrono::high_resolution_clock::now();
        kernel->execute(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Execution Time: " << elapsed << " ms\n";
        std::cout << "C[0][0] = " << C[0][0] << "\n";
    }

    return 0;
}
