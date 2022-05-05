//
//
//
//

#pragma once

#include "stdio.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "gtest/gtest.h"
#include "cogito/dnn/dnn.cuh"

#include "unittest/utils.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
struct Sigmoid {
    void operator()(T* input, T* output){
        T val = *input;
        *output = 1 / (1 + exp(-val));
    }
};


class DNN2dFixture : public testing::TestWithParam<int> 
{
public:
    void SetUp() override {

        size = GetParam();

        input_h         = static_cast<float*>(malloc(sizeof(float) * size));
        output_h        = static_cast<float*>(malloc(sizeof(float) * size));
        output_naive    = static_cast<float*>(malloc(sizeof(float) * size));

        cogito::test::initTensor(input_h, size);

        cudaMalloc(&input_d, sizeof(float) * size);
        cudaMalloc(&output_d, sizeof(float) * size);

        cudaMemcpy(input_d, input_h, sizeof(float) * size, cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        free(input_h);
        free(output_h);
        free(output_naive);

        cudaFree(input_d);
        cudaFree(output_d);
    }

protected:
    float* input_h; 
    float* input_d;
    float* output_naive;
    float* output_h;
    float* output_d;
    int size;
    cudaError_t status;
};


INSTANTIATE_TEST_SUITE_P(DNNPart,
                         DNN2dFixture,
                         testing::Values(32, 254, 256, 257, 2048));

