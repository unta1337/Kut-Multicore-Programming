// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/EY6KGQhO465FsjcSb1eBQV8BPrua8wKU8sGCkvrEgZz0Ig?e=xViTHA

// 시도한 최적화:
//  1. row -> threadIdx.x, col -> threadIdx.y로 하던 것을 반대로 row -> threadIdx.y, col -> threadIdx.x로 변경.
//  2. 각 쓰레드가 적절히 분배되어 적당한 Warp을 구성하도록 Block Dim 조정.
//  3. A와 B에 대한 Shared Memory를 하나의 변수에 할당하여 연속된 공간에 할당됨을 보장하여 메모리 뱅크 관리.
//  4. 각 쓰레드가 적절히 분배되어 서로 다른 메모리 뱅크를 참조하도록 Block Dim 조정.
//  5. A에 대한 Shared Memory에 각 쓰레드가 서로 다른 메모리 뱅크를 참조하도록 인덱싱 방법 수정.
//  6. 자주 사용하는 변수에 대한 CUDA 변수를 로컬 레지스터에 저장. (threadIdx.x 등.)
//  7. for문의 종료 조건에 대한 식을 로컬 레지스터에 저장.
//  8. 암묵적인 형변환에 대한 오버헤드를 줄이기 위해 모든 정수형 변수를 unsigned int로 통일.
//  9. 암묵적인 형변환에 대한 오버헤드를 줄이기 위해 나눗셈에 사용되는 define 상수를 float 형으로 변경.
// 10. 조건문을 간단화하여 연산 비용 감소.

#include <iostream>
#include <random>
#include <cstdlib>

#include "DS_timer.h"

#define N 2048
#define M 4096
#define L 1024

#define UNIT 16

#define F_N 2048.0f
#define F_M 4096.0f
#define F_L 1024.0f

#define F_UNIT 16.0f

const float epsilon = 1e-3;

bool is_equivalent(float a, float b) {
    return abs(a - b) < epsilon;
}

enum TIMER_NAMES {
    CPU_SERIAL,
    CPU_PARALLEL,

    GPU,
    GPU_HOST_TO_DEVICE,
    GPU_COMPUTAION,
    GPU_DEVICE_TO_HOST,

    GPU_SHARED,
    GPU_SHARED_HOST_TO_DEVICE,
    GPU_SHARED_COMPUTAION,
    GPU_SHARED_DEVICE_TO_HOST,

    NUM_TIMERS
};

__global__ void cuda_matrix_mult(float* A, float* B, float* C) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N * L)
        return;

    size_t i = index / L;
    size_t j = index % L;

    float result = 0.0f;
    for (size_t k = 0; k < M; k++)
        result += A[i * M + k] * B[k * L + j];

    C[index] = result;
}

__global__ void cuda_matrix_mult_shared(float* __A, float* __B, float* C) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int i_local = threadIdx.y;
    unsigned int j_local = threadIdx.x;

    __shared__ float mem_chunk[2][UNIT][UNIT];

    float result = 0.0f;
    unsigned int iter = (unsigned int)(F_M / F_UNIT) + 1U;

    for (unsigned int t = 0; t < iter; t++) {
        unsigned int offset = t * UNIT;

        mem_chunk[0][j_local][i_local] = __A[i * M + (j_local + offset)] * (j_local + offset < M);
        mem_chunk[1][i_local][j_local] = __B[(i_local + offset) * L + j] * (i_local + offset < M);

        __syncthreads();

        for (unsigned int k = 0; k < UNIT; k++) {
            result += mem_chunk[0][k][i_local] * mem_chunk[1][k][j_local];
        }

        __syncthreads();
    }

    if (i >= N || j >= L)
        return;

    C[i * L + j] = result;
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    DS_timer timer(NUM_TIMERS);

    timer.setTimerName(CPU_SERIAL               , (char*)"Serial                    ");
    timer.setTimerName(CPU_PARALLEL             , (char*)"Parallel                  ");

    timer.setTimerName(GPU                      , (char*)"GPU                       ");
    timer.setTimerName(GPU_HOST_TO_DEVICE       , (char*)"GPU: Host -> Device       ");
    timer.setTimerName(GPU_COMPUTAION           , (char*)"GPU: Computation          ");
    timer.setTimerName(GPU_DEVICE_TO_HOST       , (char*)"GPU: Device -> Host       ");

    timer.setTimerName(GPU_SHARED               , (char*)"GPU Shared                ");
    timer.setTimerName(GPU_SHARED_HOST_TO_DEVICE, (char*)"GPU Shared: Host -> Device");
    timer.setTimerName(GPU_SHARED_COMPUTAION    , (char*)"GPU Shared: Computation   ");
    timer.setTimerName(GPU_SHARED_DEVICE_TO_HOST, (char*)"GPU Shared: Device -> Host");

    float* A = new float[N * M];
    float* B = new float[M * L];
    float* C = new float[N * L];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            A[i * M + j] = dist(gen);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < L; j++)
            B[i * L + j] = dist(gen);

    memset(C, 0, N * L * sizeof(float));

    // Serial.
    timer.onTimer(CPU_SERIAL);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++)
            for (int k = 0; k < M; k++)
                C[i * L + j] += A[i * M + k] * B[k * L + j];

    timer.offTimer(CPU_SERIAL);

    // Parallel.
    float* C_parallel = new float[N * L];

    memset(C_parallel, 0, N * L * sizeof(float));

    timer.onTimer(CPU_PARALLEL);

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++)
            for (int k = 0; k < M; k++) {
                #pragma omp atomic
                C_parallel[i * L + j] += A[i * M + k] * B[k * L + j];
            }

    timer.offTimer(CPU_PARALLEL);

    // GPU.
    float* A_gpu, *B_gpu, *C_gpu, *temp_gpu;

    cudaMalloc(&A_gpu   , N * M * sizeof(float));
    cudaMalloc(&B_gpu   , M * L * sizeof(float));
    cudaMalloc(&temp_gpu, N * L * sizeof(float));
    C_gpu = new float[N * L];

    cudaMemset(temp_gpu, 0, N * L * sizeof(float));

    timer.onTimer(GPU);

    timer.onTimer(GPU_HOST_TO_DEVICE);
    cudaMemcpy(A_gpu, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, M * L * sizeof(float), cudaMemcpyHostToDevice);
    timer.offTimer(GPU_HOST_TO_DEVICE);

    timer.onTimer(GPU_COMPUTAION);
    size_t total_size = N * L;
    size_t unit_size = 256;

    dim3 grid_dim(ceil((float)total_size / unit_size));
    dim3 block_dim(unit_size);

    cuda_matrix_mult<<<grid_dim, block_dim>>>(A_gpu, B_gpu, temp_gpu);
    cudaDeviceSynchronize();
    timer.offTimer(GPU_COMPUTAION);

    timer.onTimer(GPU_DEVICE_TO_HOST);
    cudaMemcpy(C_gpu, temp_gpu, N * L * sizeof(float), cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_DEVICE_TO_HOST);

    timer.offTimer(GPU);

    // GPU Shared,
    float* A_gpu_shared, *B_gpu_shared, *C_gpu_shared, *temp_gpu_shared;

    cudaMalloc(&A_gpu_shared   , N * M * sizeof(float));
    cudaMalloc(&B_gpu_shared   , M * L * sizeof(float));
    cudaMalloc(&temp_gpu_shared, N * L * sizeof(float));
    C_gpu_shared = new float[N * L];

    cudaMemset(temp_gpu_shared, 0, N * L * sizeof(float));

    timer.onTimer(GPU_SHARED);

    timer.onTimer(GPU_SHARED_HOST_TO_DEVICE);
    cudaMemcpy(A_gpu_shared, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu_shared, B, M * L * sizeof(float), cudaMemcpyHostToDevice);
    timer.offTimer(GPU_SHARED_HOST_TO_DEVICE);

    timer.onTimer(GPU_SHARED_COMPUTAION);
    dim3 grid_dim_shared(ceil((float)L / UNIT), ceil((float)N / UNIT));
    dim3 block_dim_shared(UNIT, UNIT);

    cuda_matrix_mult_shared<<<grid_dim_shared, block_dim_shared>>>(A_gpu_shared, B_gpu_shared, temp_gpu_shared);
    cudaDeviceSynchronize();
    timer.offTimer(GPU_SHARED_COMPUTAION);

    timer.onTimer(GPU_SHARED_DEVICE_TO_HOST);
    cudaMemcpy(C_gpu_shared, temp_gpu_shared, N * L * sizeof(float), cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_SHARED_DEVICE_TO_HOST);

    timer.offTimer(GPU_SHARED);

    // Checking.
    bool is_correct_parallel = true;
    bool is_correct_gpu = true;
    bool is_correct_gpu_shared = true;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++)
            if (!is_equivalent(C[i * L + j], C_parallel[i * L + j])) {
                is_correct_parallel = false;
                goto loop_parallel;
            }
    loop_parallel:

    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++)
            if (!is_equivalent(C[i * L + j], C_gpu[i * L + j])) {
                is_correct_gpu = false;
                goto loop_gpu;
            }
    loop_gpu:

    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++)
            if (!is_equivalent(C[i * L + j], C_gpu_shared[i * L + j])) {
                is_correct_gpu_shared = false;
                goto loop_gpu_shared;
            }
    loop_gpu_shared:

    std::cout << "Epsilon:    " << epsilon << "\n";
    std::cout << "Parallel:   " << (is_correct_parallel   ? "Succeeded" : "Failed") << "\n";
    std::cout << "GPU:        " << (is_correct_gpu        ? "Succeeded" : "Failed") << "\n";
    std::cout << "GPU Shared: " << (is_correct_gpu_shared ? "Succeeded" : "Failed") << "\n";

    timer.printTimer();

    delete[] A, B, C;
    delete[] C_parallel, C_gpu, C_gpu_shared;

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(temp_gpu);

    cudaFree(A_gpu_shared);
    cudaFree(B_gpu_shared);
    cudaFree(temp_gpu_shared);

    return 0;
}