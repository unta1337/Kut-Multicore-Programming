// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/ET9soRnxGoZPrU37K7yhmDYBUEVGlPIZxuAmGOl4X6ZWcw?e=2qkFop

#include <iostream>
#include <sstream>
#include <vector>
#include <omp.h>

using std::cout;
using std::ostringstream;
using std::vector;

int main()
{
    const int NUM_T = omp_get_max_threads();

    vector<ostringstream> outs(NUM_T);

    #pragma omp parallel num_threads(NUM_T)
    {
        int t_num = omp_get_thread_num();
        outs[t_num] << "[Thread " << t_num << "/" << NUM_T << "] Hello OpenMP!\n";
    }

    for (auto& out : outs)
        cout << out.str();

    return 0;
}