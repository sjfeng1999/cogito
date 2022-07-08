//
//
//
//

#pragma once

namespace cogito::general::detail {

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, template<typename> class ScanOp, int ItemsPerThread>
struct ThreadScanExclusive {
public:
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ScanOpT       = ScanOp<T>;
    using ShapedTensorT = ShapedTensor<T, kItemsPerThread>;

public:
    COGITO_DEVICE
    T operator()(ShapedTensorT& output_tensor, T exclusive, const ShapedTensorT& input_tensor) { 
        ScanOpT scan_op;
        
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i) {
            output_tensor[i] = exclusive;
            exclusive = scan_op(exclusive, input_tensor[i]);
        }
        return exclusive;
    } 
};


template<typename T, template<typename> class ScanOp, int ItemsPerThread>
struct ThreadScanInclusive {
public:
    static constexpr int kItemsPerThread = ItemsPerThread;
    using ScanOpT       = ScanOp<T>;
    using ShapedTensorT = ShapedTensor<T, kItemsPerThread>;

public:
    COGITO_DEVICE
    T operator()(ShapedTensorT& output_tensor, T inclusive, const ShapedTensorT& input_tensor) { 
        ScanOpT scan_op;
        
        COGITO_PRAGMA_UNROLL
        for (int i = 0; i < kItemsPerThread; ++i) {
            inclusive = scan_op(inclusive, input_tensor[i]);
            output_tensor[i] = inclusive;
        }
        return inclusive;
    } 
};

} // namespace cogito::general::detail
