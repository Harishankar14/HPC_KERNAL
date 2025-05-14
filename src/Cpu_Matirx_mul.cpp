#include "Kernals/CpuMatrixMul.hpp"

void CpuMatrixMul::execute(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t N = A.size();
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

const char* CpuMatrixMul::name() const { return "CPU Matrix Mul"; }
const char* CpuMatrixMul::version() const { return "1.0"; }
