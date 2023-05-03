#include <stdio.h>
#include <time.h>

#include "DS_timer.h"

#define MAT_ROWS 8192
#define MAT_COLS 8192
#define GET_INDEX(row, col) (row * MAT_ROWS + col)

enum TIMER_NAMES
{
    CPU,

    GPU_22,
    GPU_22_HOST_TO_DEVICE,
    GPU_22_COMP,
    GPU_22_DEVICE_TO_HOST,

    GPU_11,
    GPU_11_HOST_TO_DEVICE,
    GPU_11_COMP,
    GPU_11_DEVICE_TO_HOST,

    GPU_21,
    GPU_21_HOST_TO_DEVICE,
    GPU_21_COMP,
    GPU_21_DEVICE_TO_HOST,

    NUM_TIMERS
};

__global__ void foo()
{
    printf("hi\n");
}

__global__ void cuda_mat_22_add(int* A, int* B)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t index = GET_INDEX(row, col);

    A[index] += B[index];
}

__global__ void cuda_mat_11_add(int* A, int* B)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    A[index] += B[index];
}

__global__ void cuda_mat_21_add(int* A, int* B)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y;

    size_t index = GET_INDEX(row, col);

    A[index] += B[index];
}

int main()
{
    srand((unsigned int)time(NULL));

    DS_timer timer(NUM_TIMERS);

    timer.setTimerName(CPU                  , (char*)"CPU                                  ");

    timer.setTimerName(GPU_22               , (char*)"GPU_22                               ");
    timer.setTimerName(GPU_22_HOST_TO_DEVICE, (char*)"GPU_22: Data transfer: Host -> Device");
    timer.setTimerName(GPU_22_COMP          , (char*)"GPU_22: Computation: Kernel          ");
    timer.setTimerName(GPU_22_DEVICE_TO_HOST, (char*)"GPU_22: Data transfer: Device -> Host");

    timer.setTimerName(GPU_11               , (char*)"GPU_11                               ");
    timer.setTimerName(GPU_11_HOST_TO_DEVICE, (char*)"GPU_11: Data transfer: Host -> Device");
    timer.setTimerName(GPU_11_COMP          , (char*)"GPU_11: Computation: Kernel          ");
    timer.setTimerName(GPU_11_DEVICE_TO_HOST, (char*)"GPU_11: Data transfer: Device -> Host");

    timer.setTimerName(GPU_21               , (char*)"GPU_21                               ");
    timer.setTimerName(GPU_21_HOST_TO_DEVICE, (char*)"GPU_21: Data transfer: Host -> Device");
    timer.setTimerName(GPU_21_COMP          , (char*)"GPU_21: Computation: Kernel          ");
    timer.setTimerName(GPU_21_DEVICE_TO_HOST, (char*)"GPU_21: Data transfer: Device -> Host");

    // Init.
    int* A, *B, *C;

    A = new int[MAT_ROWS * MAT_COLS];
    B = new int[MAT_ROWS * MAT_COLS];
    C = new int[MAT_ROWS * MAT_COLS];

    for (int row = 0; row < MAT_ROWS; row++)
    {
        for (int col = 0; col < MAT_COLS; col++)
        {
            size_t index = GET_INDEX(row, col);
            A[index] = rand() % 10;
            B[index] = rand() % 10;
        }
    }

    // Host.
    timer.onTimer(CPU);

    for (int row = 0; row < MAT_ROWS; row++)
        for (int col = 0; col < MAT_COLS; col++)
        {
            int index = GET_INDEX(row, col);
            C[index] = A[index] + B[index];
        }

    timer.offTimer(CPU);

    // Device. (22)
    int* d_22_A, *d_22_B, *d_22_C;

    cudaMalloc(&d_22_A, MAT_ROWS * MAT_COLS * sizeof(int));
    cudaMalloc(&d_22_B, MAT_ROWS * MAT_COLS * sizeof(int));
    d_22_C = new int[MAT_ROWS * MAT_COLS];

    timer.onTimer(GPU_22);

    timer.onTimer(GPU_22_HOST_TO_DEVICE);
    cudaMemcpy(d_22_A, A, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_22_B, B, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyHostToDevice);
    timer.offTimer(GPU_22_HOST_TO_DEVICE);

    timer.onTimer(GPU_22_COMP);

    dim3 d_22_grid_dim(256, 256);
    dim3 d_22_block_dim(32, 32);

    cuda_mat_22_add<<<d_22_grid_dim, d_22_block_dim>>>(d_22_A, d_22_B);
    cudaDeviceSynchronize();

    timer.offTimer(GPU_22_COMP);

    timer.onTimer(GPU_22_DEVICE_TO_HOST);
    cudaMemcpy(d_22_C, d_22_A, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_22_DEVICE_TO_HOST);

    timer.offTimer(GPU_22);

    // Device. (11)
    int* d_11_A, *d_11_B, *d_11_C;

    cudaMalloc(&d_11_A, MAT_ROWS * MAT_COLS * sizeof(int));
    cudaMalloc(&d_11_B, MAT_ROWS * MAT_COLS * sizeof(int));
    d_11_C = new int[MAT_ROWS * MAT_COLS];

    timer.onTimer(GPU_11);

    timer.onTimer(GPU_11_HOST_TO_DEVICE);
    cudaMemcpy(d_11_A, A, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_11_B, B, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyHostToDevice);
    timer.offTimer(GPU_11_HOST_TO_DEVICE);

    timer.onTimer(GPU_11_COMP);

    dim3 d_11_grid_dim(65536);
    dim3 d_11_block_dim(1024);

    cuda_mat_11_add<<<d_11_grid_dim, d_11_block_dim>>>(d_11_A, d_11_B);
    cudaDeviceSynchronize();

    timer.offTimer(GPU_11_COMP);

    timer.onTimer(GPU_11_DEVICE_TO_HOST);
    cudaMemcpy(d_11_C, d_11_A, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_11_DEVICE_TO_HOST);

    timer.offTimer(GPU_11);

    // Device. (21)
    int* d_21_A, *d_21_B, *d_21_C;

    cudaMalloc(&d_21_A, MAT_ROWS * MAT_COLS * sizeof(int));
    cudaMalloc(&d_21_B, MAT_ROWS * MAT_COLS * sizeof(int));
    d_21_C = new int[MAT_ROWS * MAT_COLS];

    timer.onTimer(GPU_21);

    timer.onTimer(GPU_21_HOST_TO_DEVICE);
    cudaMemcpy(d_21_A, A, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_21_B, B, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyHostToDevice);
    timer.offTimer(GPU_21_HOST_TO_DEVICE);

    timer.onTimer(GPU_21_COMP);

    dim3 d_21_grid_dim(8, 8192);
    dim3 d_21_block_dim(1024);

    cuda_mat_21_add<<<d_21_grid_dim, d_21_block_dim>>>(d_21_A, d_21_B);
    cudaDeviceSynchronize();

    timer.offTimer(GPU_21_COMP);

    timer.onTimer(GPU_21_DEVICE_TO_HOST);
    cudaMemcpy(d_21_C, d_21_A, MAT_ROWS * MAT_COLS * sizeof(int), cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_21_DEVICE_TO_HOST);

    timer.offTimer(GPU_21);

    // Checking.
    bool is_currect_22 = true;
    bool is_currect_11 = true;
    bool is_currect_21 = true;

    for (int row = 0; row < MAT_ROWS; row++)
    {
        for (int col = 0; col < MAT_COLS; col++)
        {
            int index = GET_INDEX(row, col);

            if (C[index] != d_22_C[index])
                is_currect_22 = false;
            if (C[index] != d_11_C[index])
                is_currect_11 = false;
            if (C[index] != d_21_C[index])
                is_currect_21 = false;
        }
    }

    printf("%s (22)\n", is_currect_22 ? "Succeeded" : "Failed");
    printf("%s (11)\n", is_currect_11 ? "Succeeded" : "Failed");
    printf("%s (21)\n", is_currect_21 ? "Succeeded" : "Failed");

    timer.printTimer();

    delete[] A, B, C;

    cudaFree(d_22_A);
    cudaFree(d_22_B);
    delete[] d_22_C;

    cudaFree(d_11_A);
    cudaFree(d_11_B);
    delete[] d_11_C;

    cudaFree(d_21_A);
    cudaFree(d_21_B);
    delete[] d_21_C;

    return 0;
}