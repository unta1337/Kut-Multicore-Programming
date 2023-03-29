#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

#define INIT_PARALLEL 1

#define NUM_T 20
#define NUM_SAMPLES (1024 * 1024 * 1024)
#define RAND_FLOAT (rand() % 10000 / 1000.0f)
#define OFFSET 32

float* inputs;
int serial_bins[10][OFFSET];
int p_lock_bins[10][OFFSET];
int p_local_lock_bins[10][OFFSET];
int p_linear_bins[10][OFFSET];
int* p_dnc_bins;

omp_lock_t g_lock;

void init_random_float();
bool is_correct(int* arr);

int main(void) {
    srand((unsigned int)time(NULL));

    inputs = (float*)malloc(NUM_SAMPLES * sizeof(float));

    DS_timer timer(6);
    timer.setTimerName(0, (char*)"Init                                                 ");
    timer.setTimerName(1, (char*)"Version 0, Serial                                    ");
    timer.setTimerName(2, (char*)"Version 1, Parallel with Lock                        ");
    timer.setTimerName(3, (char*)"Version 2, Parallel with Local bins and Lock         ");
    timer.setTimerName(4, (char*)"Version _, Parallel with Linear Gathering            ");
    timer.setTimerName(5, (char*)"Version 3, Parallel with Divide and Conqure Gathering");

    // 0. Init.
    timer.onTimer(0);

    init_random_float();

    timer.offTimer(0);

    // 1. Serial.
    timer.onTimer(1);

    for (size_t i = 0; i < NUM_SAMPLES; i++)
        serial_bins[(int)inputs[i]][0]++;

    timer.offTimer(1);

    // 2. Parallel with Lock.
    timer.onTimer(2);

    omp_init_lock(&g_lock);

    #pragma omp parallel for num_threads(NUM_T)
    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        omp_set_lock(&g_lock);
        p_lock_bins[(int)inputs[i]][0]++;
        omp_unset_lock(&g_lock);
    }

    timer.offTimer(2);

    // 3. Parallel with Local bins and Lock.
    timer.onTimer(3);

    omp_init_lock(&g_lock);

    int p_local_lock_local_bins[NUM_T][10][OFFSET];

    memset(p_local_lock_local_bins, 0, sizeof(p_local_lock_local_bins));

    #pragma omp parallel num_threads(NUM_T)
    {
        int t_num = omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < NUM_SAMPLES; i++)
            p_local_lock_local_bins[t_num][(int)inputs[i]][0]++;

        #pragma omp for
        for (int i = 0; i < NUM_T; i++) {
            for (size_t j = 0; j < 10; j++) {
                omp_set_lock(&g_lock);
                p_local_lock_bins[j][0] += p_local_lock_local_bins[i][j][0];
                omp_unset_lock(&g_lock);
            }
        }
    }

    timer.offTimer(3);

    // 4. Parallel with Linear Gathering.
    timer.onTimer(4);

    int p_linear_local_bins[NUM_T][10][OFFSET];

    memset(p_linear_local_bins, 0, sizeof(p_linear_local_bins));

    #pragma omp parallel num_threads(NUM_T)
    {
        int t_num = omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < NUM_SAMPLES; i++)
            p_linear_local_bins[t_num][(int)inputs[i]][0]++;
    }
    for (auto& local_bin : p_linear_local_bins)
        for (int i = 0; i < 10; i++)
            p_linear_bins[i][0] += local_bin[i][0];

    timer.offTimer(4);

    // 5.  Parallel with Divide and Conqure Gathering.
    timer.onTimer(5);

    int p_dnc_local_bins[NUM_T][10][OFFSET];

    memset(p_dnc_local_bins, 0, sizeof(p_dnc_local_bins));

    #pragma omp parallel num_threads(NUM_T)
    {
        int t_num = omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < NUM_SAMPLES; i++)
            p_dnc_local_bins[t_num][(int)inputs[i]][0]++;

        for (int u = 1; u < NUM_T; u *= 2) {
            for (int i = 0; i < 10; i++) {
                if (t_num % (2 * u) == 0) {
                    p_dnc_local_bins[t_num][i][0] += t_num + u < NUM_T ? p_dnc_local_bins[t_num + u][i][0] : 0;
                }
            }

            #pragma omp barrier
        }
    }

    p_dnc_bins = (int*)&p_dnc_local_bins[0];

    timer.offTimer(5);

    // Result.
    printf("1. Serial:                                     ");
    for (int i = 0; i < 10; i++)
        printf("%d: %d | ", i, serial_bins[i][0]);
    printf("is correct: %d\n", is_correct((int*)serial_bins));

    printf("2. Parallel with Lock:                         ");
    for (int i = 0; i < 10; i++)
        printf("%d: %d | ", i, p_lock_bins[i][0]);
    printf("is correct: %d\n", is_correct((int*)p_lock_bins));

    printf("3. Parallel with Local bins and Lock:          ");
    for (int i = 0; i < 10; i++)
        printf("%d: %d | ", i, p_local_lock_bins[i][0]);
    printf("is correct: %d\n", is_correct((int*)p_local_lock_bins));

    printf("4. Parallel with Linear Gathering:             ");
    for (int i = 0; i < 10; i++)
        printf("%d: %d | ", i, p_linear_bins[i][0]);
    printf("is correct: %d\n", is_correct((int*)p_linear_bins));

    printf("5. Parallel with Divide and Conqure Gathering: ");
    for (int i = 0; i < 10; i++)
        printf("%d: %d | ", i, p_dnc_bins[i * OFFSET]);
    printf("is correct: %d\n", is_correct(p_dnc_bins));

    timer.printTimer();

    return 0;
}

void init_random_float() {
#if INIT_PARALLEL
    #pragma omp parallel for num_threads(NUM_T)
#endif
    for (size_t i = 0; i < NUM_SAMPLES; i++)
        inputs[i] = RAND_FLOAT;
}

bool is_correct(int* arr) {
    for (int i = 0; i < 10; i++)
        if (arr[i * OFFSET] != serial_bins[i][0])
            return false;

    return true;
}
