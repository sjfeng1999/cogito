//
//
//
//

#pragma once

#include "stdio.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudnn.h"

#include "gtest/gtest.h"
#include "cogito/dnn/dnn.cuh"

#include "unittest/common/profiler.cuh"
#include "unittest/common/utils.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////

class DNN2dFixture : public testing::TestWithParam<std::tuple<int, int>> {
public:
    void SetUp() override {
        
        batch_size = std::get<0>(GetParam());
        inner_size = std::get<1>(GetParam());

        total_size = batch_size * inner_size;
        alpha_h      = static_cast<float*>(malloc(sizeof(float) * batch_size));
        input_h      = static_cast<float*>(malloc(sizeof(float) * total_size));
        output_h     = static_cast<float*>(malloc(sizeof(float) * total_size));
        output_naive = static_cast<float*>(malloc(sizeof(float) * total_size));

        cogito::test::initTensor(input_h, total_size);
        cogito::test::initTensor(alpha_h, batch_size);
        
        cudaMalloc(&alpha_d, sizeof(float) * batch_size);
        cudaMalloc(&input_d, sizeof(float) * total_size);
        cudaMalloc(&output_d, sizeof(float) * total_size);

        cudaMemcpy(alpha_d, alpha_h, sizeof(float) * batch_size, cudaMemcpyHostToDevice);
        cudaMemcpy(input_d, input_h, sizeof(float) * total_size, cudaMemcpyHostToDevice);

        cudnnCreate(&cudnn_handle);
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateActivationDescriptor(&act_desc);
    }

    void TearDown() override {
        free(alpha_h);
        free(input_h);
        free(output_h);
        free(output_naive);

        cudaFree(alpha_d);
        cudaFree(input_d);
        cudaFree(output_d);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyActivationDescriptor(act_desc);
        cudnnDestroy(cudnn_handle);
    }

protected:
    float *alpha_h, *alpha_d;
    float *input_h, *input_d;
    float *output_h, *output_d;
    float* output_naive;
    int batch_size, inner_size, total_size;
    float gflops;
    cudaError_t status;
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t act_desc;
    cogito::test::KernelProfiler profiler;
};


class DNN4dFixture : public testing::TestWithParam<std::tuple<int, int, int>> {
public:
    void SetUp() override {
        
        batch_size   = std::get<0>(GetParam());
        channel_size = std::get<1>(GetParam());
        tile_size    = std::get<2>(GetParam());

        total_size = batch_size * tile_size * tile_size;
        alpha_h      = static_cast<float*>(malloc(sizeof(float) * batch_size));
        input_h      = static_cast<float*>(malloc(sizeof(float) * total_size));
        output_h     = static_cast<float*>(malloc(sizeof(float) * total_size));
        output_naive = static_cast<float*>(malloc(sizeof(float) * total_size));

        cogito::test::initTensor(input_h, total_size);
        cogito::test::initTensor(alpha_h, batch_size);
        
        cudaMalloc(&alpha_d, sizeof(float) * batch_size);
        cudaMalloc(&input_d, sizeof(float) * total_size);
        cudaMalloc(&output_d, sizeof(float) * total_size);

        cudaMemcpy(alpha_d, alpha_h, sizeof(float) * batch_size, cudaMemcpyHostToDevice);
        cudaMemcpy(input_d, input_h, sizeof(float) * total_size, cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        free(alpha_h);
        free(input_h);
        free(output_h);
        free(output_naive);

        cudaFree(alpha_d);
        cudaFree(input_d);
        cudaFree(output_d);
    }

protected:
    float *alpha_h, *alpha_d;
    float *input_h, *input_d;
    float *output_h, *output_d;
    float* output_naive;
    int batch_size, channel_size, tile_size, total_size;
    float gflops;
    cudaError_t status;
    cogito::test::KernelProfiler profiler;
};


INSTANTIATE_TEST_SUITE_P(DNNPart,
                         DNN2dFixture,
                         testing::Combine(testing::Values(3, 16),
                                          testing::Values(256, 1024)));
// INSTANTIATE_TEST_SUITE_P(DNNPart,
//                          DNN4dFixture,
//                          testing::Combine(testing::Values(3, 32),
//                                           testing::Values(32, 256),
//                                           testing::Values(17, 64)));

