//
//
//
//

#pragma once

#include "stdio.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "gtest/gtest.h"
#include "cogito/blas/blas.cuh"

#include "unittest/common/profiler.cuh"
#include "unittest/common/utils.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

class BlasFixture : public testing::TestWithParam<std::tuple<int, int, int, float, float>> {
public:
    void SetUp() override {
        m = std::get<0>(GetParam());
        n = std::get<1>(GetParam());
        k = std::get<2>(GetParam());
        alpha = std::get<3>(GetParam());
        beta  = std::get<4>(GetParam());

        mn = m * n;
        mk = m * k;
        nk = n * k;

        A_h = static_cast<float*>(malloc(sizeof(float) * mk));
        B_h = static_cast<float*>(malloc(sizeof(float) * nk));
        C_h = static_cast<float*>(malloc(sizeof(float) * mn));
        res_naive = static_cast<float*>(malloc(sizeof(float) * mn));
        res_std   = static_cast<float*>(malloc(sizeof(float) * mn));

        cogito::test::initTensor(A_h, mk);
        cogito::test::initTensor(B_h, nk);
        cogito::test::initTensor(C_h, mn);

        cudaMalloc(&A_d, sizeof(float) * mk);
        cudaMalloc(&B_d, sizeof(float) * nk);
        cudaMalloc(&C_d, sizeof(float) * mn);
        cudaMalloc(&C_std, sizeof(float) * mn);

        cudaMemcpy(A_d, A_h, mk * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, nk * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(C_d, C_h, mn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(C_std, C_h, mn * sizeof(float), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        free(A_h);
        free(B_h);
        free(C_h);
        free(res_naive);
        free(res_std);

        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        cudaFree(C_std);
    }

protected:
    float *A_h, *B_h, *C_h; 
    float *A_d, *B_d, *C_d, *C_std;
    float *res_naive, *res_std;
    int m, n, k;
    int mn, mk, nk;
    float alpha, beta;
    float gflops;
    cudaError_t status;
    cogito::test::KernelProfiler profiler;
};


INSTANTIATE_TEST_SUITE_P(BlasPart,
                         BlasFixture,
                         testing::Combine(testing::Values(2048),
                                          testing::Values(2048),
                                          testing::Values(2048),
                                          testing::Values(1.2),
                                          testing::Values(0.6)));

