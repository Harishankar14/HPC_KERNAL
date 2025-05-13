#ifndef SIMD_MATRIX_MUL_HPP
#define SIMD_MATRIX_MUL_HPP

#include <omp.h>
#include "IKernel.hpp"

class SimdMatrixMul : public IKernel {
public:
    void execute(const std::vector<std::vector<int>>& A,
                 const std::vector<std::vector<int>>& B,
                 std::vector<std::vector<int>>& C) override {
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

#endif // SIMD_MATRIX_MUL_HPP
