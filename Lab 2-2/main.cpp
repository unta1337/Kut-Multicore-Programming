// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/EZYN7p1PtvZNre01V0ox7XoB4slm0wuO6Urn0OYISaQr1w?e=Ym831g

#include <stdio.h>
#include <tgmath.h>
#include <string.h>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

#define F(x) ((x) * (x))

const double ERR = 0.0001;

int main(int argc, char* argv[])
{
	DS_timer timer(2);
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Parallel");

	if (argc < 4) {
		printf("It requires three arguments\n");
		printf("Usage: %s <begin> <end> <steps>\n", argv[0]);
		return -1;
	}

    const int begin = atoi(argv[1]);
    const int end = atoi(argv[2]);
    const int steps = atoi(argv[3]);
    const double step = ((double)end - begin) / steps;

    if (step == 0)
    {
        printf("Step size if zero. Please check following:\n");
        printf("begin and end must not be the same.\n");
        printf("Your steps might be too big.\n");
        return -1;
    }

    double Y_serial = 0.0;
    double Y_parallel = 0.0;

    timer.onTimer(0);

    for (int i = 0; i < steps; i++)
    {
        double x = begin + i * step;
        Y_serial += step * (F(x) + F(x + step)) / 2;
    }

    timer.offTimer(0);

    timer.onTimer(1);

    const int num_t = omp_get_max_threads();
    double results[num_t];

    memset(results, 0, sizeof(results));

    #pragma omp parallel num_threads(num_t)
    {
        int t_num = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < steps; i++)
        {
            double x = begin + i * step;
            results[t_num] += step * (F(x) + F(x + step)) / 2;
        }
    }

    for (int i = 0; i < num_t; i++)
        Y_parallel += results[i];

    timer.offTimer(1);

    printf("[Definite integral of given function]\n");
    printf("begin: %d\n", begin);
    printf("end:   %d\n", end);
    printf("steps: %d\n", steps);
    printf("===== Result =====\n");
    printf("Serial:   %f\n", Y_serial);
    printf("Parallel: %f\n", Y_parallel);
    printf("------------------\n");
    printf("%s\n", fabs(Y_serial - Y_parallel) < ERR ? "Result matchs!" : "Result not matchs.");
    printf("allowed ERR: %f\n", ERR);

    timer.printTimer();

    return 0;
}
