// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/ET9soRnxGoZPrU37K7yhmDYBUEVGlPIZxuAmGOl4X6ZWcw?e=2qkFop

#include <iostream>
#include <omp.h>

#define NUM_T 20

int main()
{
    #pragma omp parallel num_threads(NUM_T)
    {
        std::cout << "Hello from thread " << omp_get_thread_num() << "\n";
    }

    return 0;
}