#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

#define INIT_PARALLEL 1

#define NUM_T 20
#define NUM_SAMPLES (1024 * 1024 * 1024)
#define RAND_FLOAT (rand() % 10000 / 1000.0f)

std::vector<float> inputs(NUM_SAMPLES);
std::vector<int> serial_bins(10, 0);
std::vector<int> parallel_bins(10, 0);

void init_random_float();

int main(void) {
    srand((unsigned int)time(NULL));

    DS_timer timer(3);
    timer.setTimerName(0, (char*)"Init");
    timer.setTimerName(1, (char*)"Serial");
    timer.setTimerName(2, (char*)"Parallel");

    // Init.
    timer.onTimer(0);

    init_random_float();

    timer.offTimer(0);

    // Serial.
    timer.onTimer(1);

    for (float f : inputs)
        serial_bins[static_cast<int>(std::floor(f))]++;

    timer.offTimer(1);

    // Parallel.
    timer.onTimer(2);
    std::vector<std::vector<int>> local_bins(NUM_T, std::vector<int>(10, 0));

    #pragma omp parallel num_threads(NUM_T)
    {
        int t_num = omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < NUM_SAMPLES; i++)
            local_bins[t_num][static_cast<int>(std::floor(inputs[i]))]++;
    }

    for (auto& local_bin : local_bins)
        for (int i = 0; i < 10; i++)
            parallel_bins[i] += local_bin[i];

    timer.offTimer(2);

    // Result.
    bool is_correct = true;

    for (size_t i = 0; i < 10; i++) {
        if (serial_bins[i] != parallel_bins[i]) {
            is_correct = false;
            break;
        }
    }

    std::cout << "The result is " << (is_correct ? "correct" : "incorrect.") << "\n\n";

    std::cout << "Serial Result:\n";
    for (size_t i = 0; i < 10; i++)
        std::cout << "[" << i << ", " << i + 1 << "): " << serial_bins[i] << "\n";
    std::cout << "\n";

    std::cout << "Parallel Result:\n";
    for (size_t i = 0; i < 10; i++)
        std::cout << "[" << i << ", " << i + 1 << "): " << parallel_bins[i] << "\n";

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
