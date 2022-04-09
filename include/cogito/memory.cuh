//
//
//
//
#pragma once

#include <cstdint>
#include <type_traits>

namespace cogito {

///////////////////////////////////////////////////////////////////////////////////////////////

enum class FragmentSizeType {
    kDefault,
    kLoad32B,
    kLoad64B,
    kLoad128B,
};

template<typename T, FragmentSizeType size = FragmentSizeType::kDefault>
struct FragmentSize {
    static constexpr int kSize = sizeof(T);  
};

template<typename T>
struct FragmentSize<T, FragmentSizeType::kLoad32B> {
    static constexpr int kSize = 32;  
};

template<typename T>
struct FragmentSize<T, FragmentSizeType::kLoad64B> {
    static constexpr int kSize = 64;  
};

template<typename T>
struct FragmentSize<T, FragmentSizeType::kLoad128B> {
    static constexpr int kSize = 128;  
};


template<typename T, FragmentSizeType size_type>
struct Fragment {

    static constexpr int kSize = FragmentSize<T, size_type>::kSize;
    static constexpr int kVecLength = kSize / sizeof(T);
    static_assert(kSize % sizeof(T) == 0);

    union Storage {
        int8_t storage_[kSize];
        T fragment[kVecLength];
    };

    void mov(T* tgt, T* src){
        Storage* tgt_ = reinterpret_cast<Storage*>(tgt);
        Storage* src_ = reinterpret_cast<Storage*>(src);
        *tgt_ = *src_;
    }
};

} // namespace cogito