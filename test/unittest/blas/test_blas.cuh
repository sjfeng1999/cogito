//
//
//
//

#pragma once

#include "unittest/blas/fixture_blas.cuh"

struct cublasSgemmClass {
    template<typename... Args>
    void operator()(Args... args) {
        cublasSgemm(args...);
    }
};

TEST_P(BlasFixture, GEMMTest){
    
    using GemmOpT = cogito::blas::Gemm<float, cogito::blas::MmaType::kLegacy>;
    GemmOpT()(m, n, k, alpha, A_d, k, B_d, n, beta, C_d, n);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(res_naive, 
                                      C_d, 
                                      mn * sizeof(float), 
                                      cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    gflops = (1.0 * m * n * k + m * n) / (1024 * 1024 * 1024);
    profiler.profile<GemmOpT>(gflops, m, n, k, alpha, A_d, k, B_d, n, beta, C_d, n);

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B_d, n, A_d, k, &beta, C_std, n);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(res_std, 
                                      C_std, 
                                      mn * sizeof(float), 
                                      cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    profiler.profile<cublasSgemmClass>(gflops, blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B_d, n, A_d, k, &beta, C_std, n);
    cublasDestroy(blas_handle);
    EXPECT_TRUE(cogito::test::verifyResult<float>(res_naive, res_std, mn));
};
