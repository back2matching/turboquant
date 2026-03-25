/**
 * TurboQuant CUDA Kernel — Fused rotation + quantization for KV cache compression.
 *
 * This is a custom kernel (NOT from QJL repo) that implements TurboQuant's
 * core operation: random rotation + scalar quantization, fused into one kernel
 * to minimize memory bandwidth overhead.
 *
 * Operates in FP32 to avoid the CUDA 12.1 bf16 header compatibility issue.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block size for the rotation kernel
constexpr int BLOCK_SIZE = 256;

/**
 * Fused rotation + quantization kernel.
 *
 * For each vector x of dimension D:
 *   1. Compute norm: ||x||
 *   2. Normalize: x_unit = x / ||x||
 *   3. Rotate: y = Pi @ x_unit  (Pi is D×D orthogonal)
 *   4. Quantize each coordinate of y to nearest centroid
 *   5. Store: quantized indices + norm
 *
 * @param x       Input vectors, shape (N, D), FP32
 * @param Pi_T    Rotation matrix transposed, shape (D, D), FP32  (multiply x @ Pi_T)
 * @param centroids  Codebook centroids, shape (C,), FP32
 * @param out_indices  Output indices, shape (N, D), uint8
 * @param out_norms    Output norms, shape (N,), FP32
 */
__global__ void turboquant_quantize_kernel(
    const float* __restrict__ x,
    const float* __restrict__ Pi_T,
    const float* __restrict__ centroids,
    uint8_t* __restrict__ out_indices,
    float* __restrict__ out_norms,
    int N, int D, int C
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    const float* vec = x + vec_idx * D;

    // Step 1: Compute norm
    float norm_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        norm_sq += vec[i] * vec[i];
    }

    // Warp reduction for norm
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        norm_sq += __shfl_down_sync(0xffffffff, norm_sq, offset);
    }

    // Block reduction via shared memory
    __shared__ float shared_norm[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) shared_norm[warp_id] = norm_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++) {
            total += shared_norm[i];
        }
        shared_norm[0] = sqrtf(total + 1e-10f);
    }
    __syncthreads();

    float norm = shared_norm[0];
    if (threadIdx.x == 0) out_norms[vec_idx] = norm;

    // Steps 2-4: Normalize, rotate, quantize per coordinate
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        // Rotate: y[j] = sum_i (x[i] / norm) * Pi_T[i][j]
        float y_j = 0.0f;
        for (int i = 0; i < D; i++) {
            y_j += (vec[i] / norm) * Pi_T[i * D + j];
        }

        // Quantize: find nearest centroid
        float min_dist = fabsf(y_j - centroids[0]);
        uint8_t best_idx = 0;
        for (int c = 1; c < C; c++) {
            float dist = fabsf(y_j - centroids[c]);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = (uint8_t)c;
            }
        }
        out_indices[vec_idx * D + j] = best_idx;
    }
}

/**
 * Fused dequantization + inverse rotation kernel.
 *
 * For each quantized vector:
 *   1. Look up centroids from indices
 *   2. Inverse rotate: x_hat = Pi @ y_hat  (Pi = Pi_T^T)
 *   3. Rescale by norm
 */
__global__ void turboquant_dequantize_kernel(
    const uint8_t* __restrict__ indices,
    const float* __restrict__ norms,
    const float* __restrict__ Pi,        // Rotation matrix (not transposed), D×D
    const float* __restrict__ centroids,
    float* __restrict__ out,
    int N, int D
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    float norm = norms[vec_idx];
    const uint8_t* vec_indices = indices + vec_idx * D;

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        // Inverse rotate: x_hat[j] = sum_i centroids[indices[i]] * Pi[i][j]
        float x_j = 0.0f;
        for (int i = 0; i < D; i++) {
            x_j += centroids[vec_indices[i]] * Pi[i * D + j];
        }
        out[vec_idx * D + j] = x_j * norm;
    }
}


// Python bindings
torch::Tensor turboquant_quantize_cuda(
    torch::Tensor x,           // (N, D)
    torch::Tensor Pi_T,        // (D, D)
    torch::Tensor centroids    // (C,)
) {
    auto N = x.size(0);
    auto D = x.size(1);
    auto C = centroids.size(0);

    auto x_f32 = x.to(torch::kFloat32).contiguous();
    auto Pi_T_f32 = Pi_T.to(torch::kFloat32).contiguous();
    auto centroids_f32 = centroids.to(torch::kFloat32).contiguous();

    auto indices = torch::empty({N, D}, torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
    auto norms = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    int threads = min((int)D, BLOCK_SIZE);
    turboquant_quantize_kernel<<<N, threads>>>(
        x_f32.data_ptr<float>(),
        Pi_T_f32.data_ptr<float>(),
        centroids_f32.data_ptr<float>(),
        indices.data_ptr<uint8_t>(),
        norms.data_ptr<float>(),
        N, D, C
    );

    return torch::cat({indices.to(torch::kFloat32), norms.unsqueeze(1)}, 1);
}

torch::Tensor turboquant_dequantize_cuda(
    torch::Tensor indices,     // (N, D) uint8
    torch::Tensor norms,       // (N,)
    torch::Tensor Pi,          // (D, D)
    torch::Tensor centroids    // (C,)
) {
    auto N = indices.size(0);
    auto D = indices.size(1);

    auto Pi_f32 = Pi.to(torch::kFloat32).contiguous();
    auto centroids_f32 = centroids.to(torch::kFloat32).contiguous();
    auto norms_f32 = norms.to(torch::kFloat32).contiguous();

    auto out = torch::empty({N, D}, torch::TensorOptions().dtype(torch::kFloat32).device(indices.device()));

    int threads = min((int)D, BLOCK_SIZE);
    turboquant_dequantize_kernel<<<N, threads>>>(
        indices.data_ptr<uint8_t>(),
        norms_f32.data_ptr<float>(),
        Pi_f32.data_ptr<float>(),
        centroids_f32.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &turboquant_quantize_cuda, "TurboQuant fused rotate+quantize (CUDA)");
    m.def("dequantize", &turboquant_dequantize_cuda, "TurboQuant fused dequant+rotate (CUDA)");
}
