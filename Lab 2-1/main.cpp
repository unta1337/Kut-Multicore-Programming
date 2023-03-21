// https://koreatechackr-my.sharepoint.com/:b:/g/personal/bluekds_koreatech_ac_kr/EZYN7p1PtvZNre01V0ox7XoB4slm0wuO6Urn0OYISaQr1w?e=Ym831g

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

#define PRI_RES 0

// Set the size of matrix and vector
// matrix A = m by n
// vector b = n by 1
#define m (10000)
#define n (10000)

#define GenFloat (rand() % 100 + ((float)(rand() % 100) / 100.0))
void genRandomInput();

float A[m][n];
float X[n];
float Y_serial[m];
float Y_parallel[m];

int main()
{
	DS_timer timer(2);
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Parallel");

	genRandomInput();

	//** 1. Serial code **//
	timer.onTimer(0);

	//** HERE
	//** Write your code implementing the serial algorithm here
    for (size_t i = 0; i < m ; i++) {
        for (size_t j = 0; j < n ; j++) {
            Y_serial[i] += A[i][j] * X[j];
        }
    }

	timer.offTimer(0);

	//** 2. Parallel code **//
	timer.onTimer(1);

	//** HERE
	//** Write your code implementing the parallel algorithm here
    const int num_t = omp_get_max_threads();

    #pragma omp parallel num_threads(num_t)
    {
        int t_num = omp_get_thread_num();
        for (size_t i = t_num; i < m ; i += num_t) {
            for (size_t j = 0; j < n; j++) {
                Y_parallel[i] += A[i][j] * X[j];
            }
        }
    }

	timer.offTimer(1);

	//** 3. Result checking code **//
	bool isCorrect = true;

	//** HERE
	//** Wriet your code that compares results of serial and parallel algorithm
	// Set the flag 'isCorrect' to true when they are matched
    for (size_t i = 0; i < n; i++) {
        if (Y_parallel[i] != Y_serial[i]) {
            isCorrect = false;
            break;
        }
    }

	if (!isCorrect)
		printf("Results are not matched :(\n");
	else
		printf("Results are matched! :)\n");

	timer.printTimer();

	EXIT_WIHT_KEYPRESS;
}

void genRandomInput(void) {
	// A matrix
	LOOP_INDEX(row, m) {
		LOOP_INDEX(col, n) {
			A[row][col] = GenFloat;
		}
	}

	LOOP_I(n)
		X[i] = GenFloat;

	memset(Y_serial, 1, sizeof(float) * m);
	memset(Y_parallel, 0, sizeof(float) * m);
}
