// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/ET9soRnxGoZPrU37K7yhmDYBUEVGlPIZxuAmGOl4X6ZWcw?e=2qkFop

#include <iostream>
#include <omp.h>

int main()
{
    const int NUM_T = omp_get_max_threads();

    #pragma omp parallel num_threads(NUM_T)
    {
        std::cout << "[Thread " << omp_get_thread_num() << "/" << NUM_T << "] Hello OpenMP!\n";
    }

    return 0;
}
