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

#include "unittest/utils.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////


class BlasFixture : public testing::TestWithParam<std::tuple<int, int, int, float, float>> 
{
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

        // cogito::test::initTensor(input_h, size);

        cudaMalloc(&A_d, sizeof(float) * mk);
        cudaMalloc(&B_d, sizeof(float) * nk);
        cudaMalloc(&C_d, sizeof(float) * mn);

        // cudaMemcpy(input_d, input_h, m * n * sizeof(float), cudaMemcpyHostToDevice);
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
    }

protected:
    float *A_h, *B_h, *C_h; 
    float *A_d, *B_d, *C_d;
    float *res_naive, *res_std;
    int m;
    int n;
    int k;
    int mn, mk, nk;
    float alpha, beta;
    cudaError_t status;
};


INSTANTIATE_TEST_SUITE_P(BlasPart,
                         BlasFixture,
                         testing::Combine(testing::Values(256, 1024),
                                          testing::Values(256, 1024),
                                          testing::Values(256, 1024),
                                          testing::Values(1.0, 0.5),
                                          testing::Values(1.0, 2.0)));

