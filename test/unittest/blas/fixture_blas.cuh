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

class BlasFixture : public testing::TestWithParam<std::tuple<int, int, int, nv_half, nv_half>> {
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

        A_h = static_cast<nv_half*>(malloc(sizeof(nv_half) * mk));
        B_h = static_cast<nv_half*>(malloc(sizeof(nv_half) * nk));
        C_h = static_cast<nv_half*>(malloc(sizeof(nv_half) * mn));
        res_naive = static_cast<nv_half*>(malloc(sizeof(nv_half) * mn));
        res_std   = static_cast<nv_half*>(malloc(sizeof(nv_half) * mn));

        cogito::test::initTensor(A_h, mk);
        cogito::test::initTensor(B_h, nk);
        cogito::test::initTensor(C_h, mn);

        cudaMalloc(&A_d, sizeof(nv_half) * mk);
        cudaMalloc(&B_d, sizeof(nv_half) * nk);
        cudaMalloc(&C_d, sizeof(nv_half) * mn);
        cudaMalloc(&C_std, sizeof(nv_half) * mn);

        cudaMemcpy(A_d, A_h, mk * sizeof(nv_half), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, nk * sizeof(nv_half), cudaMemcpyHostToDevice);
        cudaMemcpy(C_d, C_h, mn * sizeof(nv_half), cudaMemcpyHostToDevice);
        cudaMemcpy(C_std, C_h, mn * sizeof(nv_half), cudaMemcpyHostToDevice);
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
    nv_half *A_h, *B_h, *C_h; 
    nv_half *A_d, *B_d, *C_d, *C_std;
    nv_half *res_naive, *res_std;
    int m, n, k;
    int mn, mk, nk;
    nv_half alpha, beta;
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

