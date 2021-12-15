#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <crt/device_functions.h>
#include <time.h>


void print_elapsed(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.4fs\n", elapsed);
}

__global__ void sort(int* d_a, int step, int stage, int N, int doPrint) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int seqL = pow(2, step);		// sequnce length per stage
	int n = seqL / (pow(2, stage - 1));
	int shift = n / 2;

	if (idx < N) {
		if (doPrint == 1) printf("thread %d is active.\n", idx);

		if ((idx % n) < shift) { //decide if thread is active or not

			// decide if the sort is ascending or descending 
			/* even | ascending + */
			if ((idx / seqL) % 2 == 0 && idx < N) {
				if (doPrint == 1) printf("(+) ascending | even: has been reached by thread %d, seqL = %d.\n", idx, seqL);
				if (d_a[idx] > d_a[idx + shift]) {
					int temp = d_a[idx];
					d_a[idx] = d_a[idx + shift];
					d_a[idx + shift] = temp;
					if (doPrint == 1) printf("(+) ascending | swaped d_a[%d] = %d , for d_a[%d] = %d\n", idx, d_a[idx], idx + shift, d_a[idx + shift]);
				}
				else {
					if (doPrint == 1) printf("(+) ascending | thread %d did not swap\n", idx);
				}
			}

			/* odd | descending - */
			if ((idx / seqL) % 2 == 1 && idx < N) {
				if (doPrint == 1) printf("(-) descending | odd: has been reached by thread %d, seqL = %d.\n", idx, seqL);
				if (d_a[idx] < d_a[idx + shift]) {
					int temp = d_a[idx];
					d_a[idx] = d_a[idx + shift];
					d_a[idx + shift] = temp;
					if (doPrint == 1) printf("(-) descending | swaped d_a[%d] = %d , for d_a[%d] = %d\n", idx, d_a[idx], idx + shift, d_a[idx + shift]);
				}
				else {
					if (doPrint == 1) printf("(-) descending | thread %d did not swap\n", idx);
				}
			}
		}
		else {
			if (doPrint == 1) printf("thread %d is inactive.\n", idx);
		}
	}
}

void random_ints(int* x, int size)
{
	int i;
	for (i = 0; i < size; i++) {
		x[i] = rand() % 5000;
	}
}

void print_values(int* a, int* b, int N) {
	printf("Before    After\n");
	for (int i = 0; i < N; i++) {
		printf("a[%d]=%d , b[%d]=%d\n", i, a[i], i, b[i]);
	}
}

void bitonic_sort(int N, int blocks, int threads, int* d_a, int doPrint) {

	int itters = log2(N);
	int totalSteps = itters;

	for (int i = 1; i <= totalSteps; i++) {
		if(doPrint == 1) printf("step %d now.\n", i);
		int totalStages = i;		// calculate stages for the current step

		for (int j = 1; j <= totalStages; j++) {
			if (doPrint == 1) printf("stage %d now.\n", j);

			sort << <blocks, threads >> > (d_a, i, j, N, doPrint); // N/2 threads
		}
	}

}

int take_input() {

	int values;
	printf("Please enter the number of valuse:");
	scanf("%d", &values);
	int exp = ceil(log(values) / log(2));
	int N = pow(2, exp);
	int padding = N - values;

	return N;
}

int print_state() {
	int doPrint;
	printf("Do you want to print logs(1= Yes, 0= No)? \n");
	printf("Please note disabling logging will have better performance! \n");
	scanf("%d", &doPrint);
	if (doPrint == 0) return doPrint;
	if (doPrint == 1) return doPrint;
	while (!(doPrint == 0 || doPrint == 1)) {
		printf("Please enter value 1 or 0 (1= Yes, 0= No)\n");
		scanf("%d", &doPrint);
		if (doPrint == 0) return doPrint;
		if (doPrint == 1) return doPrint;
	}
}

void select_device(){
	int select;
	printf("Do you want to select a device(1= Yes, 0= No)? \n");
	printf("If No default is chosen. \n");
	scanf("%d", &select);
	if (select == 1) {
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			printf("Device Number: %d\n", i);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n",
				prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		}
		printf("Please enter the device number: \n");
		int chosen = scanf("%d", &chosen);
		cudaSetDevice(chosen);
	}
}

int main()
{
	clock_t start, stop;

	int doPrint = print_state();

	select_device();

	int N = take_input();


	int* a, * b;
	int* d_a;
	int size = N * sizeof(int);

	cudaMalloc((void**)&d_a, size);

	a = (int*)malloc(size); random_ints(a, N);
	b = (int*)malloc(size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	int device;
	cudaGetDevice(&device);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	int MaxThreads = props.maxThreadsPerBlock;
	int threads = (int &)MaxThreads;
	int blocks = N / threads + 1;
	printf("Running on: threads=%d , blocks=%d\n", threads,blocks);

	start = clock();
	bitonic_sort(N, blocks, threads, d_a, doPrint);
	stop = clock();

	cudaMemcpy(b, d_a, size, cudaMemcpyDeviceToHost);

	print_values(a, b, N);

	print_elapsed(start, stop);


	free(a);	free(b);
	cudaFree(d_a);

	return 0;
		
	}
