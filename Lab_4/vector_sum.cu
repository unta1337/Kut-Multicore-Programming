// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/EdbpTtPwDpxPg9pZ7axFsH8BXqPY3_Z-G4uXIjdjo-RoUw?e=8PPMOW

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "DS_timer.h"

enum TIMER_NAMES
{
    CPU,
    GPU,
    GPU_HOST_TO_DEVICE,
    GPU_COMP,
    GPU_DEVICE_TO_HOST,
    NUM_TIMERS
};

__global__ void cuda_vec_add(int* v, int* u)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    v[index] += u[index];
}

int main()
{
    srand((unsigned int)time(NULL));

    DS_timer timer(NUM_TIMERS);

    timer.setTimerName(CPU               , (char*)"CPU                               ");
    timer.setTimerName(GPU               , (char*)"GPU                               ");
    timer.setTimerName(GPU_HOST_TO_DEVICE, (char*)"GPU: Data transfer: Host -> Device");
    timer.setTimerName(GPU_COMP          , (char*)"GPU: Computation: Kernel          ");
    timer.setTimerName(GPU_DEVICE_TO_HOST, (char*)"GPU: Data transfer: Device -> Host");

    const size_t size_unit = 1024;
    const size_t size_factor = 1 << 1;
    const size_t vec_size = size_unit * size_unit * size_factor;

    const size_t block_size = size_unit;
    const size_t grid_size = vec_size / block_size;
    const size_t total_num_threads = block_size * grid_size;

    assert(vec_size % block_size == 0 && "vec_size must be divisible by block_size.");

    printf("==== CUDA info ====\n");
    printf("*only uses x-dim in both grid and block.\n");
    printf("total_num_threads: %zu * %zu * %zu (%zu)\n", size_unit, size_unit, size_factor, total_num_threads);
    printf("block_size: %zu\n", block_size);
    printf("grid_size: %zu\n", grid_size);
    printf("===================\n\n");

    dim3 grid_dim(grid_size);
    dim3 block_dim(block_size);

    // Init.
    int* p, *q, *r;

    p = new int[vec_size];
    q = new int[vec_size];
    r = new int[vec_size];

    for (int i = 0; i < vec_size; i++)
    {
        p[i] = rand() % 10;
        q[i] = rand() % 10;
    }

    // Host.
    timer.onTimer(CPU);

    for (int i = 0; i < vec_size; i++)
        r[i] = p[i] + q[i];

    timer.offTimer(CPU);

    // Device.
    int* d_p, *d_q, *d_r;

    cudaMalloc(&d_p, vec_size * sizeof(int));
    cudaMalloc(&d_q, vec_size * sizeof(int));
    d_r = new int[vec_size];

    timer.onTimer(GPU);

    timer.onTimer(GPU_HOST_TO_DEVICE);
    cudaMemcpy(d_p, p, vec_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, vec_size * sizeof(int), cudaMemcpyHostToDevice);
    timer.offTimer(GPU_HOST_TO_DEVICE);

    timer.onTimer(GPU_COMP);
    cuda_vec_add<<<grid_dim, block_dim>>>(d_p, d_q);
    cudaDeviceSynchronize();
    timer.offTimer(GPU_COMP);

    timer.onTimer(GPU_DEVICE_TO_HOST);
    cudaMemcpy(d_r, d_p, vec_size * sizeof(int), cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_DEVICE_TO_HOST);

    timer.offTimer(GPU);

    // Checking.
    bool is_currect = true;
    for (int i = 0; i < vec_size; i++)
        if (r[i] != d_r[i])
        {
            is_currect = false;
            break;
        }

    printf("%s\n", is_currect ? "Succeeded" : "Failed");

    timer.printTimer();

    delete[] p, q, r, d_r;
    cudaFree(d_p);
    cudaFree(d_q);

    return 0;
}