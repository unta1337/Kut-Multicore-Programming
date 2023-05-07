#include <iostream>
#include <random>
#include <cstdlib>

#include "DS_timer.h"

enum TIMER_NAMES {
    CPU_SERIAL,
    CPU_PARALLEL,
    GPU,
    GPU_HOST_TO_DEVICE,
    GPU_COMPUTAION,
    GPU_DEVICE_TO_HOST,
    NUM_TIMERS
};

__global__ void cuda_matrix_mult(float* A, float* B, float* C, int N, int M, int L) {
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    DS_timer timer(NUM_TIMERS);

    timer.setTimerName(CPU_SERIAL        , (char*)"Serial             ");
    timer.setTimerName(CPU_PARALLEL      , (char*)"Parallel           ");

    timer.setTimerName(GPU               , (char*)"GPU                ");
    timer.setTimerName(GPU_HOST_TO_DEVICE, (char*)"GPU: Host -> Device");
    timer.setTimerName(GPU_COMPUTAION    , (char*)"GPU: Computation   ");
    timer.setTimerName(GPU_DEVICE_TO_HOST, (char*)"GPU: Device -> Host");

    const int N = 1024;
    const int M = 512;
    const int L = 2048;

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
                C[i * L + j] += A[i * M + k] + B[k * L + j];

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
                C_parallel[i * L + j] += A[i * M + k] + B[k * L + j];
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
    cuda_matrix_mult<<<1, 1>>>(A_gpu, B_gpu, temp_gpu, N, M, L);
    cudaDeviceSynchronize();
    timer.offTimer(GPU_COMPUTAION);

    timer.onTimer(GPU_DEVICE_TO_HOST);
    cudaMemcpy(C_gpu, temp_gpu, N * L * sizeof(float), cudaMemcpyHostToDevice);
    timer.offTimer(GPU_DEVICE_TO_HOST);

    timer.offTimer(GPU);

    // Checking.
    bool is_correct_parallel = true;
    bool is_correct_gpu = true;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++)
            if (C[i * L + j] != C_parallel[i * L + j]) {
                is_correct_parallel = false;
                goto loop_parallel;
            }
    loop_parallel:

    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++)
            if (C[i * L + j] != C_gpu[i * L + j]) {
                is_correct_gpu = false;
                goto loop_gpu;
            }
    loop_gpu:

    std::cout << "Parallel: " << (is_correct_parallel ? "Succeeded" : "Failed") << "\n";
    std::cout << "GPU:      " << (is_correct_gpu      ? "Succeeded" : "Failed") << "\n";

    timer.printTimer();

    delete[] A, B, C;
    delete[] C_parallel, C_gpu;

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(temp_gpu);

    return 0;
}