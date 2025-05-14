#include "Kernals/GpuMatrixMul.hpp"

void GpuMatrixMul::execute(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t N = A.size();
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

const char* GpuMatrixMul::name() const { return "GPU Matrix Mul (Simulated)"; }
const char* GpuMatrixMul::version() const { return "1.1"; }
