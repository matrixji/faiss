/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <faiss/gpu/utils/DeviceDefs.cuh>

namespace faiss {
namespace gpu {

// defines to simplify the SASS assembly structure file/line in the profiler
#if CUDA_VERSION >= 9000 && !defined(__HIP_PLATFORM_HCC__)
#define SHFL_SYNC(VAL, SRC_LANE, WIDTH) \
    __shfl_sync(0xffffffff, VAL, SRC_LANE, WIDTH)
#else
#define SHFL_SYNC(VAL, SRC_LANE, WIDTH) __shfl(VAL, SRC_LANE, WIDTH)
#endif

template <typename T>
inline __device__ T shfl(const T val, int srcLane, int width = kWarpSize) {
#if CUDA_VERSION >= 9000 && !defined(__HIP_PLATFORM_HCC__)
    return __shfl_sync(0xffffffff, val, srcLane, width);
#else
    return __shfl(val, srcLane, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl(T* const val, int srcLane, int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl(v, srcLane, width);
}

template <typename T>
inline __device__ T
shfl_up(const T val, unsigned int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000 && !defined(__HIP_PLATFORM_HCC__)
    return __shfl_up_sync(0xffffffff, val, delta, width);
#else
    return __shfl_up(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_up(
        T* const val,
        unsigned int delta,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl_up(v, delta, width);
}

template <typename T>
inline __device__ T
shfl_down(const T val, unsigned int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000 && !defined(__HIP_PLATFORM_HCC__)
    return __shfl_down_sync(0xffffffff, val, delta, width);
#else
    return __shfl_down(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_down(
        T* const val,
        unsigned int delta,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_down(v, delta, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000 && !defined(__HIP_PLATFORM_HCC__)
    return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
    return __shfl_xor(val, laneMask, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_xor(
        T* const val,
        int laneMask,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_xor(v, laneMask, width);
}

// CUDA 9.0+ has half shuffle
#if CUDA_VERSION < 9000 || defined(__HIP_PLATFORM_HCC__)
inline __device__ half shfl(half v, int srcLane, int width = kWarpSize) {
#ifndef __HIP_PLATFORM_HCC__
    unsigned int vu = v.x;
#else
    unsigned int vu = __half_as_short(v);
#endif
    vu = __shfl(vu, srcLane, width);

    half h;
#ifndef __HIP_PLATFORM_HCC__
    h.x = (unsigned short)vu;
#else
    h = __short_as_half((unsigned short)vu);
#endif
    return h;
}

inline __device__ half shfl_xor(half v, int laneMask, int width = kWarpSize) {
#ifndef __HIP_PLATFORM_HCC__
    unsigned int vu = v.x;
#else
    unsigned int vu = __half_as_short(v);
#endif
    vu = __shfl_xor(vu, laneMask, width);

    half h;
#ifndef __HIP_PLATFORM_HCC__
    h.x = (unsigned short)vu;
#else
    h = __short_as_half((unsigned short)vu);
#endif
    return h;
}
#endif // CUDA_VERSION

} // namespace gpu
} // namespace faiss
