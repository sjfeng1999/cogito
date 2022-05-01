//
//
//
//

#pragma once 

#include "unittest/blas/fixture_blas.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(BlasFixture, GEMMTest){
    
    status = cogito::blas::Gemm<float, cogito::blas::MmaType::kLegacy>()(m, n, k, 
                                                                         alpha, 
                                                                         A_d, k, 
                                                                         B_d, n, 
                                                                         beta, 
                                                                         C_d, n);
    // EXPECT_EQ(cudaSuccess, status);
    // EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, 
    //                                   output_d, 
    //                                   m * n * sizeof(float), 
    //                                   cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // cublasHandle_t blas_handle;
    // cublasCreate(&blas_handle);
    // cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc);
    // cublasDestroy(blas_handle);

    // EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, size));
};


