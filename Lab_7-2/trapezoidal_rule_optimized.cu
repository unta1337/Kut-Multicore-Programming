#include <stdio.h>
#include <omp.h>

#include "DS_timer.h"

#define F(x) ((x) * (x))

const double epsilon = 1e-3;

bool is_equivalent(double a, double b) {
    return abs(a - b) < epsilon;
}

enum TIMER_NAMES {
    CPU_SERIAL,
    CPU_PARALLEL,

    GPU,
    GPU_HOST_TO_DEVICE,
    GPU_COMPUTAION,
    GPU_DEVICE_TO_HOST,

    NUM_TIMERS
};

__global__ void calc_definite_integral(int a, int b, int n, double* I_chunks) {
    size_t beta = gridDim.x;
    size_t alpha = n / beta;

    size_t block_begin = alpha * blockIdx.x;
    bool is_end = blockIdx.x == beta - 1;
    size_t block_end = !is_end * (alpha * (blockIdx.x + 1)) +
                        is_end * n;

    double dx = (double)(b - a) / n;
    __shared__ double local_sum[1024];

    for (size_t i = block_begin + threadIdx.x; i < block_end; i += blockDim.x) {
        double x = a + dx * i;
        local_sum[threadIdx.x] += (F(x) + F(x + dx)) * dx / 2;
    }

    __syncthreads();

    for (size_t stride = 1; stride < blockDim.x; stride <<= 1) {
		if (threadIdx.x % (2 * stride) == 0)
			local_sum[threadIdx.x] += local_sum[threadIdx.x + stride];
		__syncthreads();
	}

    if (threadIdx.x == 0)
        I_chunks[blockIdx.x] += local_sum[0];
}

int main(int argc, char* argv[]) {
    int a, b, n;

    if (argc == 1) {
        a = 0;
        b = 1024;
        n = 1073741824;
    } else if (argc == 4) {
        a = strtol(argv[1], NULL, 10);
        b = strtol(argv[2], NULL, 10);
        n = strtol(argv[3], NULL, 10);
    } else {
        fprintf(stderr, "Argument not enough: a b n.\n");
        return 1;
    }

    DS_timer timer(NUM_TIMERS);

    timer.setTimerName(CPU_SERIAL        , (char*)"Serial             ");
    timer.setTimerName(CPU_PARALLEL      , (char*)"Parallel           ");

    timer.setTimerName(GPU               , (char*)"GPU                ");
    timer.setTimerName(GPU_HOST_TO_DEVICE, (char*)"GPU: Host -> Device");
    timer.setTimerName(GPU_COMPUTAION    , (char*)"GPU: Computation   ");
    timer.setTimerName(GPU_DEVICE_TO_HOST, (char*)"GPU: Device -> Host");

    // Serial.
    timer.onTimer(CPU_SERIAL);

    double dx = (double)(b - a) / n;

    double I = 0.0f;
    for (size_t i = 0; i < n; i++) {
        double x = a + dx * i;
        I += (F(x) + F(x + dx)) * dx / 2;
    }

    timer.offTimer(CPU_SERIAL);

    // Parallel.
    timer.onTimer(CPU_PARALLEL);

    double p_dx = (double)(b - a) / n;
    double p_I = 0.0;

    #pragma omp parallel reduction(+ : p_I)
    {
        size_t index = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < n; i++) {
            double x = a + p_dx * i;
            p_I += (F(x) + F(x + p_dx)) * p_dx / 2;
        }
    }

    timer.offTimer(CPU_PARALLEL);

    // GPU.
    // Setup.
    size_t num_chunks = 20;
    size_t unit = 1024;

    double d_I = 0.0;

    double* I_chunks = new double[num_chunks];
    double* d_I_chunks;
    cudaMalloc(&d_I_chunks, num_chunks * sizeof(double));

    timer.onTimer(GPU);

    timer.onTimer(GPU_HOST_TO_DEVICE);
    cudaMemset(d_I_chunks, 0, num_chunks * sizeof(double));
    timer.offTimer(GPU_HOST_TO_DEVICE);

    timer.onTimer(GPU_COMPUTAION);
    calc_definite_integral<<<num_chunks, unit>>>(a, b, n, d_I_chunks);
    cudaDeviceSynchronize();
    timer.offTimer(GPU_COMPUTAION);

    timer.onTimer(GPU_DEVICE_TO_HOST);
    cudaMemcpy(I_chunks, d_I_chunks, num_chunks * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_chunks; i++)
        d_I += I_chunks[i];
    timer.offTimer(GPU_DEVICE_TO_HOST);

    timer.offTimer(GPU);

    printf("(a, b): (%d, %d)\n", a, b);
    printf("n: %d\n\n", n);

    printf("CPU Serial  : %f\n",     I);
    printf("CPU Parallel: %f\n",   p_I);
    printf("GPU:          %f\n\n", d_I);

    printf("CPU Parallel: %s.\n", is_equivalent(I, p_I) ? "Succeded" : "Failed");
    printf("GPU         : %s.\n", is_equivalent(I, d_I) ? "Succeded" : "Failed");

    timer.printTimer();

    return 0;
}
