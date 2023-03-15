#include <omp.h>
#include "common.h"
#include "../DSTimer/DS_timer.h"

#define MANUAL_DIST

#define DECRYPT_ALGORITHM(a,b,c) (c += a*50 + b)
 
uchar* readData(char* _fileName, int _dataSize);
bool writeData(char* _fileName, int _dataSize, uchar* _data);

// Lab 1-2
int main(int argc, char** argv)
{
	DS_timer timer(2, 1);

	timer.setTimerName(0, (char*)"Serial Algorithm");
	timer.setTimerName(1, (char*)"Parallel Algorithm");

	if (argc < 6) {
		printf("It requires five arguments\n");
		printf("Usage: Extuction_file inputImgA inputImgB Width Height OutputFileName\n");
		return -1;
	}

	int width = atoi(argv[3]);
	int height = atoi(argv[4]);
	int dataSize = width * height * 3;

	// Read input data
	uchar* A = readData(argv[1], dataSize);
	uchar* B = readData(argv[2], dataSize);

	uchar* serialC = new uchar[dataSize];
	uchar* parallelC = new uchar[dataSize]; 
	memset(serialC, 0, sizeof(uchar) * dataSize);
	memset(parallelC, 0, sizeof(uchar) * dataSize);

	// Decrypt the image
	// The algorith is defined as DECRYPT_ALGORITHM(a,b,c)
	// See the definition at the top of this source code

	// 1. Serial algorithm
	timer.onTimer(0);
	for (int i = 0; i < dataSize; i++) {
		DECRYPT_ALGORITHM(A[i], B[i], serialC[i]);
	}
	timer.offTimer(0);

	timer.onTimer(1);
	// ***************************************************************
	// Wirte the decyprt result to parallelC array
	// Hint: DECRYPT_ALGORITHM(A[i], B[i], parallelC[i])

	// 2. Parallel algorithm
	const int num_t = 20;							// 사용할 쓰레드 개수.

#ifdef MANUAL_DIST
	// 방법 1:
	// 수동으로 데이터 분배.
	// 
	// 이미지는 R * C의 2차원으로 구성되므로, 각 쓰레드마다 일정 개수의 행을 부여.
	// e.g.
	//      6000 * 4000 이미지, 20개의 쓰레드 -> 각 쓰레드가 6000 / 20 = 300개의 행을 담당.
	//      각 픽셀의 정보가 RGB로 세 개씩이므로 각 쓰레드가 300 * 4000 * 3개의 데이터를 담당.

	// 담당 행 분배.
	const int unit_row_count = height / num_t;		// 한 개의 쓰레드가 담당할 행의 개수.

	// 구간 분배.
	int t_begin[num_t];
	int t_end[num_t];

	for (int i = 0; i < num_t; i++)
	{
		t_begin[i] = unit_row_count * width * 3 * i;
		t_end[i] = unit_row_count * width * 3 * (i + 1) - 1;
	}

	// 행을 분배한 후, 남은 부분은 마지막 쓰레드가 담당.
	t_end[num_t - 1] = dataSize - 1;

	printf("[Workload Balance]\n");
	for (int i = 0; i < num_t; i++)
		printf("[Thread %d/%d] %.2f %%\n", i, num_t, ((double)t_end[i] - t_begin[i] + 1) / dataSize * 100);

	#pragma omp parallel num_threads(num_t)
	{
		int t_num = omp_get_thread_num();
		for (int i = t_begin[t_num]; i <= t_end[t_num]; i++) {
			DECRYPT_ALGORITHM(A[i], B[i], parallelC[i]);
		}
	}
#endif

#ifndef MANUAL_DIST
	// 방법 2:
	// 자동으로 데이터 분배.
	// omp parallel for 사용.

	#pragma omp parallel for num_threads(num_t)
	for (int i = 0; i < dataSize; i++) {
		DECRYPT_ALGORITHM(A[i], B[i], parallelC[i]);
	}
#endif

	// **************************************************************
	timer.offTimer(1);

	// Check the results
	bool isCorrect = true;
	for (int i = 0; i < dataSize; i++) {
		if (serialC[i] != parallelC[i]) {
			isCorrect = false; 
			break;
		}
	}

	if (isCorrect)
		printf("The results is correct - Good job!\n");
	else
		printf("The result is not correct! :(\n");

	printf("Your computer has %d logical cores\n", omp_get_num_procs());
	timer.printTimer();


	if (!writeData(argv[5], dataSize, parallelC))
		printf("Fail to write the data\n");
	else
		printf("The decrption result was written to %s\n", argv[5]);
}

uchar* readData(char* _fileName, int _dataSize)
{
	uchar* data;
	data = new uchar[_dataSize];
	memset(data, 0, sizeof(uchar) * _dataSize);

	FILE* fp = NULL;
	fopen_s(&fp, _fileName, "rb");
	if (!fp) {
		printf("Fail to open %s\n", _fileName);
		return NULL;
	}

	fread(data, sizeof(uchar), _dataSize, fp);
	fclose(fp);

	return data;
}

bool writeData(char* _fileName, int _dataSize, uchar* _data)
{
	FILE* fp = NULL;
	fopen_s(&fp, _fileName, "wb");
	if (!fp) {
		printf("Fail to open %s\n", _fileName);
		return false;
	}

	fwrite(_data, sizeof(uchar), _dataSize, fp);
	fclose(fp);

	return true;
}