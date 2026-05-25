
#include "cuda_common.h"
#include "brainevent/common.h"

// ============================================================================
// Bit extraction macros (same as bitpack_binary_fcnmv.cu)
// ============================================================================

#define IS_ACTIVE_PACKED(packed, idx) \
    ((__ldg(&(packed)[(idx) >> 5]) >> ((idx) & 31)) & 1)

#define IS_ACTIVE_PACKED_SMEM(smem, idx) \
    (((smem)[(idx) >> 5] >> ((idx) & 31)) & 1)


// 2D compact scatter tile:
//   blockIdx.x spans groups of 10 active rows.
//   blockIdx.y spans 32-connection chunks inside each row.
//   Each warp handles one active row and one 32-wide connection chunk.
// The block sorts the 10x32 target indices in shared memory, groups equal
// adjacent keys, and emits one atomic add per key group.
#define COMPACT_FCNMV_TILE_ROWS 10
#define COMPACT_FCNMV_TILE_COLS 32
#define COMPACT_FCNMV_SORT_SIZE 512
#define COMPACT_FCNMV_SENTINEL 2147483647

// --- Scatter 2D-and-atomic homo (compact, block-sort/group prototype) ---
#define DEFINE_CS_2D_AND_ATOMIC_HOMO(SUFFIX, WEIGHT_T, ACC_T, READ_W, ATOMIC_ADD_W)    \
__global__ void _cs_2d_and_atomic_homo_kern##SUFFIX(                                   \
    const int32_t*  __restrict__ indices,                                               \
    const int32_t*  __restrict__ active_ids,                                            \
    const int32_t*  __restrict__ n_active_ptr,                                          \
    WEIGHT_T*       __restrict__ output,                                                \
    const WEIGHT_T* __restrict__ weights,                                               \
    int n_conn                                                                          \
) {                                                                                     \
    __shared__ int32_t s_keys[COMPACT_FCNMV_SORT_SIZE];                                 \
    int warp_in_block = threadIdx.x >> 5;                                               \
    int lane = threadIdx.x & 31;                                                        \
    int local = threadIdx.x;                                                            \
    int na = __ldg(n_active_ptr);                                                       \
    int active_row = blockIdx.x * COMPACT_FCNMV_TILE_ROWS + warp_in_block;              \
    int k = blockIdx.y * COMPACT_FCNMV_TILE_COLS + lane;                                \
    int key = COMPACT_FCNMV_SENTINEL;                                                   \
    if (warp_in_block < COMPACT_FCNMV_TILE_ROWS && active_row < na && k < n_conn) {      \
        int row = __ldg(&active_ids[active_row]);                                       \
        const int32_t* i_row = indices + (size_t)row * n_conn;                          \
        key = __ldg(&i_row[k]);                                                         \
    }                                                                                   \
    if (local < COMPACT_FCNMV_TILE_ROWS * COMPACT_FCNMV_TILE_COLS)                      \
        s_keys[local] = key;                                                            \
    for (int i = local + blockDim.x; i < COMPACT_FCNMV_SORT_SIZE; i += blockDim.x)       \
        s_keys[i] = COMPACT_FCNMV_SENTINEL;                                             \
    __syncthreads();                                                                    \
    for (int size = 2; size <= COMPACT_FCNMV_SORT_SIZE; size <<= 1) {                   \
        for (int stride = size >> 1; stride > 0; stride >>= 1) {                        \
            for (int i = local; i < COMPACT_FCNMV_SORT_SIZE; i += blockDim.x) {         \
                int ixj = i ^ stride;                                                   \
                if (ixj > i) {                                                          \
                    int a = s_keys[i];                                                  \
                    int b = s_keys[ixj];                                                \
                    bool ascending = ((i & size) == 0);                                 \
                    if ((ascending && a > b) || (!ascending && a < b)) {                \
                        s_keys[i] = b;                                                  \
                        s_keys[ixj] = a;                                                \
                    }                                                                   \
                }                                                                       \
            }                                                                           \
            __syncthreads();                                                            \
        }                                                                               \
    }                                                                                   \
    ACC_T w0 = READ_W(__ldg(&weights[0]));                                              \
    for (int i = local; i < COMPACT_FCNMV_SORT_SIZE; i += blockDim.x) {                 \
        int sorted_key = s_keys[i];                                                     \
        if (sorted_key == COMPACT_FCNMV_SENTINEL) continue;                             \
        if (i > 0 && sorted_key == s_keys[i - 1]) continue;                             \
        int count = 1;                                                                  \
        for (int j = i + 1; j < COMPACT_FCNMV_SORT_SIZE && s_keys[j] == sorted_key; ++j) \
            ++count;                                                                    \
        ATOMIC_ADD_W(&output[sorted_key], w0 * (ACC_T)count);                           \
    }                                                                                   \
}

// ---- FFI macro: scatter homo compact 2d_and_atomic backend ----
#define FFI_CS_2D_AND_ATOMIC_HOMO(SUFFIX, WEIGHT_C_T)                                         \
void compact_binary_fcnmv_scatter_2d_and_atomic_homo##SUFFIX(                                 \
    const BE::Tensor weights, const BE::Tensor indices,                                       \
    const BE::Tensor packed, const BE::Tensor active_ids,                                     \
    const BE::Tensor n_active, BE::Tensor output,                                             \
    int64_t stream                                                                            \
) {                                                                                           \
    (void)packed;                                                                             \
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);                                  \
    int n_orig = static_cast<int>(active_ids.size(0));                                        \
    int n_conn = static_cast<int>(indices.size(1));                                           \
    int n_post = static_cast<int>(output.size(0));                                            \
    const WEIGHT_C_T* d_w    = static_cast<const WEIGHT_C_T*>(weights.data_ptr());            \
    const int32_t*    d_idx  = static_cast<const int32_t*>(indices.data_ptr());               \
    const int32_t*    d_aids = static_cast<const int32_t*>(active_ids.data_ptr());            \
    const int32_t*    d_na   = static_cast<const int32_t*>(n_active.data_ptr());              \
    WEIGHT_C_T*       d_out  = static_cast<WEIGHT_C_T*>(output.data_ptr());                   \
    cudaMemsetAsync(d_out, 0, (size_t)n_post * sizeof(WEIGHT_C_T), s);                        \
    if (n_orig == 0) return;                                                                  \
    dim3 block(COMPACT_FCNMV_TILE_ROWS * 32);                                                 \
    dim3 grid(                                                                                \
        (n_orig + COMPACT_FCNMV_TILE_ROWS - 1) / COMPACT_FCNMV_TILE_ROWS,                     \
        (n_conn + COMPACT_FCNMV_TILE_COLS - 1) / COMPACT_FCNMV_TILE_COLS);                    \
    _cs_2d_and_atomic_homo_kern##SUFFIX<<<grid, block, 0, s>>>(                               \
        d_idx, d_aids, d_na, d_out, d_w, n_conn);                                             \
    BE_CHECK_KERNEL_LAUNCH();                                                                 \
}

DEFINE_CS_2D_AND_ATOMIC_HOMO(_f32, float, float, READ_F32, atomicAdd)

// @BE compact_binary_fcnmv_scatter_2d_and_atomic_homo_f32
FFI_CS_2D_AND_ATOMIC_HOMO(_f32, float)
