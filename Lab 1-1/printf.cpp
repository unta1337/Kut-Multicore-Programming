// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/ET9soRnxGoZPrU37K7yhmDYBUEVGlPIZxuAmGOl4X6ZWcw?e=2qkFop

#include <stdio.h>
#include <omp.h>

#define NUM_T 20

int main()
{
    #pragma omp parallel num_threads(NUM_T)
    {
        printf("Hello from thread %d\n", omp_get_thread_num());
    }

    return 0;
}