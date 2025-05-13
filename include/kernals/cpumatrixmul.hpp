#ifndef CPU_MATRIX_MUL_HPP
#define CPU_MATRIX_MUL_HPP

#include "IKernel.hpp"

class CpuMatrixMul : public IKernel {
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

    const char* name() const override { return "CPU Matrix Mul"; }
    const char* version() const override { return "1.0"; }
};

#endif // CPU_MATRIX_MUL_HPP
