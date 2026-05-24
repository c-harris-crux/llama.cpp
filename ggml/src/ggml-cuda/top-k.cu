#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_USE_HIP
static constexpr int GGML_HIP_TOP_K_MAX_K = 128;
static constexpr int GGML_HIP_TOP_K_BLOCK_SIZE = 256;

static __device__ __forceinline__ bool top_k_pair_better(
        const float a_val, const int a_idx, const float b_val, const int b_idx) {
    return a_val > b_val || (a_val == b_val && a_idx < b_idx);
}

static __device__ __forceinline__ bool top_k_pair_before(
        const float val, const int idx, const float prev_val, const int prev_idx) {
    return val < prev_val || (val == prev_val && idx > prev_idx);
}

static __global__ void top_k_f32_i32_hip(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int ncols,
        const int k) {
    const int row = blockIdx.x;
    const float * src_row = src + row * ncols;
    int * dst_row = dst + row * k;

    __shared__ float vals[GGML_HIP_TOP_K_BLOCK_SIZE];
    __shared__ int idxs[GGML_HIP_TOP_K_BLOCK_SIZE];
    __shared__ float prev_val_s;
    __shared__ int prev_idx_s;

    if (threadIdx.x == 0) {
        prev_val_s = INFINITY;
        prev_idx_s = -1;
    }
    __syncthreads();

    for (int out = 0; out < k; ++out) {
        float best_val = -INFINITY;
        int best_idx = ncols;

        const float prev_val = prev_val_s;
        const int prev_idx = prev_idx_s;

        for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
            const float val = src_row[col];
            if (top_k_pair_before(val, col, prev_val, prev_idx) &&
                    top_k_pair_better(val, col, best_val, best_idx)) {
                best_val = val;
                best_idx = col;
            }
        }

        vals[threadIdx.x] = best_val;
        idxs[threadIdx.x] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && top_k_pair_better(vals[threadIdx.x + stride], idxs[threadIdx.x + stride],
                                                          vals[threadIdx.x],          idxs[threadIdx.x])) {
                vals[threadIdx.x] = vals[threadIdx.x + stride];
                idxs[threadIdx.x] = idxs[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            dst_row[out] = idxs[0];
            prev_val_s = vals[0];
            prev_idx_s = idxs[0];
        }
        __syncthreads();
    }
}
#endif // GGML_USE_HIP

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef GGML_USE_HIP
    GGML_ASSERT(k <= GGML_HIP_TOP_K_MAX_K);
    const dim3 block_dims(GGML_HIP_TOP_K_BLOCK_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    top_k_f32_i32_hip<<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols, k);
    GGML_UNUSED(pool);
#else
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
#endif
}
