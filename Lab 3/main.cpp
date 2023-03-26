#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

#define INIT_PARALLEL 1

#define NUM_T 20
#define NUM_SAMPLES (1024 * 1024)
// #define NUM_SAMPLES (1024 * 1024 * 1024)
#define RAND_FLOAT (rand() % 10000 / 1000.0f)

std::vector<float> inputs(NUM_SAMPLES);
std::vector<int> serial_bins(10, 0);
std::vector<int> p_lock_bins(10, 0);
std::vector<int> p_local_lock_bins(10, 0);
std::vector<int> p_linear_bins(10, 0);
std::vector<int> p_dnc_bins(10, 0);

omp_lock_t g_lock;

void init_random_float();

int main(void) {
    srand((unsigned int)time(NULL));

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

    for (float f : inputs)
        serial_bins[static_cast<int>(f)]++;

    timer.offTimer(1);

    // 2. Parallel with Lock.
    timer.onTimer(2);

    omp_init_lock(&g_lock);

    #pragma omp parallel for num_threads(NUM_T)
    for (size_t i = 0; i < NUM_SAMPLES; i++)
    {
        omp_set_lock(&g_lock);
        p_lock_bins[static_cast<int>(inputs[i])]++;
        omp_unset_lock(&g_lock);
    }

    timer.offTimer(2);

    // 3. Parallel with Local bins and Lock.
    timer.onTimer(3);

    omp_init_lock(&g_lock);

    std::vector<std::vector<int>> p_local_lock_local_bins(NUM_T, std::vector<int>(10, 0));

    #pragma omp parallel num_threads(NUM_T)
    {
        int t_num = omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < NUM_SAMPLES; i++)
            p_local_lock_local_bins[t_num][static_cast<int>(inputs[i])]++;

        #pragma omp for
        for (int i = 0; i < NUM_T; i++) {
            for (size_t j = 0; j < 10; j++) {
                omp_set_lock(&g_lock);
                p_local_lock_bins[j] += p_local_lock_local_bins[i][j];
                omp_unset_lock(&g_lock);
            }
        }
    }

    timer.offTimer(3);

    // 4. Parallel with Linear Gathering.
    timer.onTimer(4);

    std::vector<std::vector<int>> p_linear_local_bins(NUM_T, std::vector<int>(10, 0));

    #pragma omp parallel num_threads(NUM_T)
    {
        int t_num = omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < NUM_SAMPLES; i++)
            p_linear_local_bins[t_num][static_cast<int>(inputs[i])]++;
    }
    for (auto& local_bin : p_linear_local_bins)
        for (int i = 0; i < 10; i++)
            p_linear_bins[i] += local_bin[i];

    timer.offTimer(4);

    // 5.  Parallel with Divide and Conqure Gathering.
    timer.onTimer(5);
    timer.offTimer(5);

    // Result.
    std::cout << "1. Serial:                                      ";
    for (int i = 0; i < 10; i++)
        std::cout << i << ": " << serial_bins[i] << " | ";
    std::cout << "\n";

    std::cout << "2. Parallel with Lock:                          ";
    for (int i = 0; i < 10; i++)
        std::cout << i << ": " << p_lock_bins[i] << " | ";
    std::cout << "\n";

    std::cout << "3. Parallel with Local bins and Lock:           ";
    for (int i = 0; i < 10; i++)
        std::cout << i << ": " << p_local_lock_bins[i] << " | ";
    std::cout << "\n";

    std::cout << "4. Parallel with Linear Gathering:              ";
    for (int i = 0; i < 10; i++)
        std::cout << i << ": " << p_linear_bins[i] << " | ";
    std::cout << "\n";

    std::cout << "5. Parallel with Divide and Conqure Gatheriung: ";
    for (int i = 0; i < 10; i++)
        std::cout << i << ": " << p_dnc_bins[i] << " | ";
    std::cout << "\n";

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
