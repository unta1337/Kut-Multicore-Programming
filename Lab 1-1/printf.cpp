// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/ET9soRnxGoZPrU37K7yhmDYBUEVGlPIZxuAmGOl4X6ZWcw?e=2qkFop

#include <stdio.h>
#include <omp.h>

int main()
{
    const int NUM_T = omp_get_max_threads();

    #pragma omp parallel num_threads(NUM_T)
    {
        printf("[Thread %d/%d] Hello OpenMP!\n", omp_get_thread_num(), NUM_T);
    }

    return 0;
}