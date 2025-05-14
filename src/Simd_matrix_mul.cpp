#include "Kernals/SimdMatrixMul.hpp"
#include <omp.h>

void SimdMatrixMul::execute(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t N = A.size();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(N); ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

const char* SimdMatrixMul::name() const { return "SIMD (OpenMP) Matrix Mul"; }
const char* SimdMatrixMul::version() const { return "2.0"; }
