/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK(x) do {                                      \
    cudaError_t err = x;                                   \
    if (err != cudaSuccess) {                              \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",     \
            __FILE__, __LINE__, cudaGetErrorString(err));  \
        std::exit(1);                                      \
    }                                                      \
} while (0)

__global__ void normalize_kernel(int ny, int nx, const float* data, float* norm) {
    int y = blockIdx.x;
    if (y >= ny) return;

    float sum = 0.0f;
    for (int x = threadIdx.x; x < nx; x += blockDim.x) {
        sum += data[x + y * nx];
    }

    __shared__ float ssum[256];
    ssum[threadIdx.x] = sum;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            ssum[threadIdx.x] += ssum[threadIdx.x + step];
        }
        __syncthreads();
    }

    float mean = ssum[0] / nx;

    float sq = 0.0f;
    for (int x = threadIdx.x; x < nx; x += blockDim.x) {
        float v = data[x + y * nx] - mean;
        sq += v * v;
    }

    __shared__ float ssq[256];
    ssq[threadIdx.x] = sq;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            ssq[threadIdx.x] += ssq[threadIdx.x + step];
        }
        __syncthreads();
    }

    float inv_norm = rsqrtf(ssq[0]);

    for (int x = threadIdx.x; x < nx; x += blockDim.x) {
        norm[x + y * nx] = (data[x + y * nx] - mean) * inv_norm;
    }
}

__global__ void correlate_kernel(int ny, int nx, const float* norm, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j >= ny || j > i) return;

    float sum = 0.0f;
    for (int x = 0; x < nx; ++x) {
        sum += norm[x + i * nx] * norm[x + j * nx];
    }

    result[i + j * ny] = sum;
}

void correlate(int ny, int nx, const float* data, float* result) {
    float* data_gpu = nullptr;
    float* norm_gpu = nullptr;
    float* result_gpu = nullptr;

    CHECK(cudaMalloc(&data_gpu, ny * nx * sizeof(float)));
    CHECK(cudaMalloc(&norm_gpu, ny * nx * sizeof(float)));
    CHECK(cudaMalloc(&result_gpu, ny * ny * sizeof(float)));

    CHECK(cudaMemcpy(data_gpu, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    normalize_kernel<<<ny, 256>>>(ny, nx, data_gpu, norm_gpu);
    CHECK(cudaGetLastError());

    CHECK(cudaMemset(result_gpu, 0, ny * ny * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((ny + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    correlate_kernel<<<grid, block>>>(ny, nx, norm_gpu, result_gpu);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, result_gpu, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(data_gpu));
    CHECK(cudaFree(norm_gpu));
    CHECK(cudaFree(result_gpu));
}
