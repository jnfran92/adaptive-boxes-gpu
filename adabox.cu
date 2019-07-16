
#include "stdio.h"
#include "stdlib.h"

// getters
#include "./include/getters.h"

//data
#include "./data/squares.h"


__device__ int sum_numbers(int a, int b){
	return a+b;
}

__global__ void find_largest_rectangle(long m, long n, int *data_matrix){

	const int coords_m = 5;
	const int coords_n = 4;

	__shared__ int coords[coords_m * coords_n];

	int i = threadIdx.y;
	int j = threadIdx.x;


	int g_i = blockDim.y * blockIdx.y + i;
	int g_j = blockDim.x * blockIdx.x + j;


	printf("i %d  j %d   gi %d   gj %d \n",i,j,g_i,g_j);

	if (i==0){
		get_right_bottom_rectangle(0, 4, m, n, data_matrix);
	}

	if (i==1){
		get_right_top_rectangle(0, 4, n, data_matrix);
	}

	if (i==2){
		get_left_bottom_rectangle(0, 4, m, n, data_matrix);
	}

	if (i==3){
		get_left_top_rectangle(0, 4, n, data_matrix);
	}

	/*coords[i*coords_n + 0] = */
}


int main(){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	printf("m %ld , n% ld\n",m, n);

	
	get_right_bottom_rectangle(0, 4, m, n, data);
	get_left_bottom_rectangle(0, 4, m, n, data);
	get_left_top_rectangle(0, 4, n, data);
	get_right_top_rectangle(0, 4, n, data);

	// CUDA
	int *data_d;

	// Get Mem
	cudaMalloc((void **)&data_d, sizeof(int)*m*n );

	// Copy to device
	cudaMemcpy(data_d, data, sizeof(int)*m*n, cudaMemcpyHostToDevice);


	dim3 grid(50, 50, 1);
	dim3 block(1, 4, 1); // fixed size
	
	
	find_largest_rectangle<<<grid, block>>>(m, n, data_d);
	cudaDeviceSynchronize();


	return 0;
}
