// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef BRAINEVENT_CUDA_TRAITS_CUH
#define BRAINEVENT_CUDA_TRAITS_CUH
// CUDA-only scalar operations shared by native launchers and BrainEvent bindings.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace cuda_traits {
enum class ValueType : int { f16, bf16, f32, f64 };
#ifdef __CUDACC__
/**
 * Full-warp unsigned sum, returned to every lane.
 *
 * Uses the sm_80+ ``__reduce_add_sync`` instruction and a butterfly shuffle
 * reduction on older architectures. All 32 lanes must participate.
 */
__device__ __inline__ unsigned warp_reduce_add_u32(unsigned value) {
#if __CUDA_ARCH__ >= 800
    return __reduce_add_sync(0xffffffffU, value);
#else
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    return value;
#endif
}

__device__ __inline__ void atomic_add_f32(float* addr, float val) {
    atomicAdd(addr, val);
}

/**
 * Atomic add for float64.
 *
 * Native atomic operation supported on all CUDA architectures.
 *
 * @param addr  Pointer to memory location
 * @param val   Value to add
 */
__device__ __inline__ void atomic_add_f64(double* addr, double val) {
    atomicAdd(addr, val);
}

/**
 * Atomic add for float16.
 *
 * Uses native atomicAdd on sm_70+ (Volta and newer).
 * Falls back to CAS-based emulation on older architectures.
 *
 * PRECONDITION (pre-sm_70 emulation path only): @p addr must lie within a
 * buffer whose enclosing 4-byte word is fully owned by the allocation, i.e.
 * half buffers must be 4-byte aligned and even-length (padded).  The emulation
 * rounds @p addr down to a 4-byte boundary and performs a 32-bit atomicCAS,
 * which reads AND writes the adjacent 2 bytes; for the last element of an
 * odd-length buffer at an allocation boundary that is a 2-byte out-of-bounds
 * read-modify-write and can corrupt a neighbouring allocation (M16).  On
 * sm_70+ the native path is used and this restriction does not apply.
 *
 * @param addr  Pointer to memory location
 * @param val   Value to add (float32, will be converted to float16)
 */
__device__ __inline__ void atomic_add_f16(__half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    // Emulate with CAS on older architectures.
    // WARNING: touches the neighbouring 2 bytes - see the M16 precondition above
    // (half buffers must be 4-byte aligned and even-length / padded).
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __half2float(*reinterpret_cast<__half*>(&h));
        __half new_h = __float2half(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}

/**
 * Atomic add for bfloat16.
 *
 * Uses native atomicAdd on sm_80+ (Ampere and newer).
 * Falls back to CAS-based emulation on older architectures.
 *
 * PRECONDITION (pre-sm_80 emulation path only): same as atomic_add_f16 - @p addr
 * must lie in a 4-byte-aligned, even-length (padded) bf16 buffer.  The 32-bit
 * atomicCAS reads AND writes the adjacent 2 bytes, so the last element of an
 * odd-length buffer at an allocation boundary is a 2-byte out-of-bounds
 * read-modify-write (M16).  On sm_80+ the native path is used.
 *
 * @param addr  Pointer to memory location
 * @param val   Value to add (float32, will be converted to bfloat16)
 */
__device__ __inline__ void atomic_add_bf16(__nv_bfloat16* addr, float val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, __float2bfloat16(val));
#else
    // Emulate with CAS on older architectures.
    // WARNING: touches the neighbouring 2 bytes - see the M16 precondition above
    // (bf16 buffers must be 4-byte aligned and even-length / padded).
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~(size_t)2
    );
    int shift = ((reinterpret_cast<size_t>(addr) & 2) != 0) ? 16 : 0;
    unsigned int assumed, old_val = *base, updated;
    do {
        assumed = old_val;
        unsigned short h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
        float cur = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&h));
        __nv_bfloat16 new_h = __float2bfloat16(cur + val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_h);
        updated = (assumed & ~(0xFFFFu << shift)) | (static_cast<unsigned int>(new_us) << shift);
        old_val = atomicCAS(base, assumed, updated);
    } while (assumed != old_val);
#endif
}


#endif // __CUDACC__
template<class T> struct ValueTraits;
template<> struct ValueTraits<float> {
    using AccT = float;
    static constexpr ValueType type = ValueType::f32;
    __host__ __device__ static AccT read(float value) { return value; }
    __host__ __device__ static float write(AccT value) { return value; }
#ifdef __CUDACC__
    __device__ static void atomic_add(float* ptr, AccT value) { cuda_traits::atomic_add_f32(ptr, value); }
#endif
};
template<> struct ValueTraits<double> {
    using AccT = double;
    static constexpr ValueType type = ValueType::f64;
    __host__ __device__ static AccT read(double value) { return value; }
    __host__ __device__ static double write(AccT value) { return value; }
#ifdef __CUDACC__
    __device__ static void atomic_add(double* ptr, AccT value) { cuda_traits::atomic_add_f64(ptr, value); }
#endif
};
template<> struct ValueTraits<__half> {
    using AccT = float;
    static constexpr ValueType type = ValueType::f16;
    __host__ __device__ static AccT read(__half value) { return __half2float(value); }
    __host__ __device__ static __half write(AccT value) { return __float2half(value); }
#ifdef __CUDACC__
    __device__ static void atomic_add(__half* ptr, AccT value) { cuda_traits::atomic_add_f16(ptr, value); }
#endif
};
template<> struct ValueTraits<__nv_bfloat16> {
    using AccT = float;
    static constexpr ValueType type = ValueType::bf16;
    __host__ __device__ static AccT read(__nv_bfloat16 value) { return __bfloat162float(value); }
    __host__ __device__ static __nv_bfloat16 write(AccT value) { return __float2bfloat16(value); }
#ifdef __CUDACC__
    __device__ static void atomic_add(__nv_bfloat16* ptr, AccT value) { cuda_traits::atomic_add_bf16(ptr, value); }
#endif
};
template<class T> struct EventTraits {
    __host__ __device__ static bool active(T value) { return value != T(0); }
};
template<class ValueT> struct TileConfig {
    using AccT = typename ValueTraits<ValueT>::AccT;
    static constexpr int width = sizeof(AccT) == 8 ? 12288 : 24576;
    static constexpr int shared_bytes = width * sizeof(AccT);
};
} // namespace cuda_traits
#endif
