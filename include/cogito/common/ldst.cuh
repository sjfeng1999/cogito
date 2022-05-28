//
//
// 
//

#pragma once 

#include "cogito/cogito.cuh"
#include "cogito/tensor.cuh"

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int blockSize>
struct ThreadLoad {
public:
    static constexpr int kBlockSize     = blockSize;
    static constexpr int kItemPerThread = kBlockSize;

public:
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemPerThread>& tensor, const T* ptr, bool valid) {
        if (valid) {
            tensor.load<0, kItemPerThread>(ptr);
        }
    }
};

template<typename T, int blockSize>
struct ThreadStore {
public:
    static constexpr int kBlockSize     = blockSize;
    static constexpr int kItemPerThread = kBlockSize;

public:
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemPerThread>& tensor, T* ptr, bool valid) {
        if (valid) {
            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < kBlockSize; ++i) {
                *(ptr + i) = tensor[i];
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int blockSize, int stripSize = 1>
struct WarpLoad {
public:
    static constexpr int kBlockSize     = blockSize;
    static constexpr int kStripSize     = stripSize;
    static constexpr int kWarpLineSize  = cogito::kWarpSize * kBlockSize;
    static constexpr int kItemPerThread = kBlockSize * kStripSize;

public:
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemPerThread>& tensor, const T* ptr, bool valid) {
        if (valid) {
            int tid = threadIdx.x;

            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < kStripSize; ++i) {
                COGITO_PRAGMA_UNROLL
                for (int j = 0; j < kBlockSize; ++j) {
                    tensor[i * kStripSize + j] = ptr[i * kWarpLineSize + tid * kBlockSize + j];
                }
            }
        }
    }
};

template<typename T, int blockSize, int stripSize = 1>
struct WarpStore {
public:
    static constexpr int kBlockSize     = blockSize;
    static constexpr int kStripSize     = stripSize;
    static constexpr int kWarpLineSize  = cogito::kWarpSize * kBlockSize;
    static constexpr int kItemPerThread = kBlockSize * kStripSize;

public:
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemPerThread>& tensor, T* ptr, bool valid) {
        if (valid) {
            int tid = threadIdx.x;

            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < kStripSize; ++i) {
                COGITO_PRAGMA_UNROLL
                for (int j = 0; j < kBlockSize; ++j) {
                    ptr[i * kWarpLineSize + tid * kBlockSize + j] = tensor[i * kStripSize + j];
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int blockDimX, int blockSize, int stripSize = 1>
struct BlockLoad {
public:
    static constexpr int kBlockDim      = blockDimX;
    static constexpr int kBlockSize     = blockSize;
    static constexpr int kStripSize     = stripSize;
    static constexpr int kBlockLineSize = kBlockDim * kBlockSize;
    static constexpr int kItemPerThread = kBlockSize * kStripSize;

public:
    COGITO_DEVICE
    static void load(ShapedTensor<T, kItemPerThread>& tensor, const T* ptr, bool valid) {
        if (valid) {
            int tid = threadIdx.x;

            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < kStripSize; ++i) {
                COGITO_PRAGMA_UNROLL
                for (int j = 0; j < kBlockSize; ++j) {
                    tensor[i * kStripSize + j] = ptr[i * kBlockLineSize + tid * kBlockSize + j];
                }
            }
        }
    }
};

template<typename T, int blockDimX, int blockSize, int stripSize = 1>
struct BlockStore {
public:
    static constexpr int kBlockDim      = blockDimX;
    static constexpr int kBlockSize     = blockSize;
    static constexpr int kStripSize     = stripSize;
    static constexpr int kBlockLineSize = kBlockDim * kBlockSize;
    static constexpr int kItemPerThread = kBlockSize * kStripSize;

public:
    COGITO_DEVICE
    static void store(const ShapedTensor<T, kItemPerThread>& tensor, T* ptr, bool valid) {
        if (valid) {
            int tid = threadIdx.x;

            COGITO_PRAGMA_UNROLL
            for (int i = 0; i < kStripSize; ++i) {
                COGITO_PRAGMA_UNROLL
                for (int j = 0; j < kBlockSize; ++j) {
                    ptr[i * kBlockLineSize + tid * kBlockSize + j] = tensor[i * kStripSize + j];
                }
            }
        }
    }
};

} // namespace cogito

