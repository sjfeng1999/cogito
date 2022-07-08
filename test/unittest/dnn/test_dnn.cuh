//
//
//
//

#pragma once

#include "unittest/dnn/fixture_dnn.cuh"


struct cudnnSoftmaxClass {
    template<typename... Args>
    void operator()(Args... args) {
        cudnnSoftmaxForward(args...);
    }
};

TEST_P(DNN2dFixture, SigmoidTest){
    using TestSigmoid = cogito::dnn::Sigmoid<float>;
    TestSigmoid()(batch_size, inner_size, input_d, output_d);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, output_d, total_size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    float alpha_ = 1;
    float beta_  = 0;

    EXPECT_EQ(CUDNN_STATUS_SUCCESS, cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, inner_size)); 
    EXPECT_EQ(CUDNN_STATUS_SUCCESS, cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, inner_size)); 
    EXPECT_EQ(CUDNN_STATUS_SUCCESS, cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0)); 
    EXPECT_EQ(CUDNN_STATUS_SUCCESS, cudnnActivationForward(cudnn_handle, act_desc, 
                                        &alpha_, input_desc, input_d, &beta_, output_desc, output_d));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_naive, output_d, total_size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, total_size));
};

TEST_P(DNN2dFixture, PReluTest) {
    using TestPRelu = cogito::dnn::PRelu<float>;
    TestPRelu()(batch_size, inner_size, input_d, alpha_d, output_d);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, output_d, total_size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    gflops = 1.0 * total_size / (1024 * 1024 * 1024);
    profiler.profile<TestPRelu>(gflops, batch_size, inner_size, input_d, alpha_d, output_d);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < inner_size; ++j) {
            output_naive[i * inner_size + j] = 
                input_h[i * inner_size + j] > 0 ? input_h[i * inner_size + j] : alpha_h[i] * input_h[i * inner_size + j];
        }
    }
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, total_size));
}

TEST_P(DNN2dFixture, SoftmaxTest){
    using TestSoftmax = cogito::dnn::Softmax<float>;
    TestSoftmax()(batch_size, inner_size, input_d, output_d);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_h, output_d, total_size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));

    gflops = total_size * 4.0 / (1024 * 1024 * 1024);
    profiler.profile<TestSoftmax>(gflops, batch_size, inner_size, input_d, output_d);

    float alpha_ = 1;
    float beta_  = 0;

    EXPECT_EQ(CUDNN_STATUS_SUCCESS, cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, inner_size, 1, 1)); 
    EXPECT_EQ(CUDNN_STATUS_SUCCESS, cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, inner_size, 1, 1)); 
    EXPECT_EQ(CUDNN_STATUS_SUCCESS, cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, 
                                        &alpha_, input_desc, input_d, &beta_, output_desc, output_d));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output_naive, output_d, total_size * sizeof(float), 
                                      cudaMemcpyDeviceToHost));
    EXPECT_TRUE(cogito::test::verifyResult<float>(output_h, output_naive, total_size));

    profiler.profile<cudnnSoftmaxClass>(gflops, cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, 
                                        &alpha_, input_desc, input_d, &beta_, output_desc, output_d);
};
