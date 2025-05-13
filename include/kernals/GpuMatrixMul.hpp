#ifndef GPU_MATRIX_MUL_HPP
#define GPU_MATRIX_MUL_HPP

#include "IKernel.hpp"

class GpuMatrixMul : public IKernel {
public:
    void execute(const std::vector<std::vector<int>>& A,
                 const std::vector<std::vector<int>>& B,
                 std::vector<std::vector<int>>& C) override {
        size_t N = A.size();
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                for (size_t k = 0; k < N; ++k)
                    C[i][j] += A[i][k] * B[k][j];
    }

    const char* name() const override { return "GPU Matrix Mul (Simulated)"; }
    const char* version() const override { return "1.1"; }
};

#endif // GPU_MATRIX_MUL_HPP
