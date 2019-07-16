
#include "stdio.h"
#include "stdlib.h"

// getters
#include "./include/getters.h"

//data
#include "./data/squares.h"

// GPU kernel
__global__ void find_largest_rectangle(int idx_i, int idx_j, long m, long n, int *data_matrix, int *areas){

	const int coords_m = 5;
	const int coords_n = 4;

	__shared__ int coords[coords_m * coords_n];

	int i = threadIdx.y;
	int j = threadIdx.x;


	int g_i = blockDim.y * blockIdx.y + i;
	int g_j = blockDim.x * blockIdx.x + j;

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = blockDim.y;


	/*printf("i %d  j %d   gi %d   gj %d \n",i,j,g_i,g_j);*/
	
	int results[4] = {};

	if (i==0){
		get_right_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results);
	}

	if (i==1){
		get_right_top_rectangle(idx_i, idx_j, n, data_matrix, results);
	}

	if (i==2){
		get_left_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results);
	}

	if (i==3){
		get_left_top_rectangle(idx_i, idx_j, n, data_matrix, results);
	}

	/*printf("x1 %d    x2 %d   y1 %d   y2 %d   \n", results[0],  results[1],  results[2],  results[3]);*/
	coords[i*coords_n + 0] = results[0];
	coords[i*coords_n + 1] = results[1];
	coords[i*coords_n + 2] = results[2];
	coords[i*coords_n + 3] = results[3];

	__syncthreads();


	if (i==0){
		int a = coords[2*coords_n + 1];
		int b = coords[3*coords_n + 1];
		int pl = a;
		if (b > a){
			pl = b;
		}
		coords[4*coords_n + i] = pl;
	}

	if (i==1){
		int a = coords[0*coords_n + 1];
		int b = coords[1*coords_n + 1];
		int pr = a;
		if (b < a){
			pr = b;
		}
		coords[4*coords_n + i] = pr;
	}

	if (i==2){
		int a = coords[1*coords_n + 3];
		int b = coords[3*coords_n + 3];
		int pt = a;
		if (b > a){
			pt = b;
		}
		coords[4*coords_n + i] = pt;
	}

	if (i==3){
		int a = coords[0*coords_n + 3];
		int b = coords[2*coords_n + 3];
		int pb = a;
		if (b < a){
			pb = b;
		}
		coords[4*coords_n + i] = pb;
	}

	__syncthreads();

	printf("final x1 %d,    x2 %d,    y1 %d,    y2 %d \n",coords[4*coords_n + 0], coords[4*coords_n + 1],coords[4*coords_n + 2] ,coords[4*coords_n + 3]);

	// get area
	if (i==0){
		int a = abs( coords[coords_n*4 + 0] -  coords[coords_n*4 + 1] );
		int b = abs( coords[coords_n*4 + 2] -  coords[coords_n*4 + 3] );
		int area = a*b;
		printf("area %d\n", area);
	 	coords[coords_n*0 + 0] = area; // Saving area in coords[0][0]
	       	areas[b_i*b_n + b_j] = area;
		printf("bi %d    bj %d\n", b_i,b_j);
	}
	__syncthreads();

	// get max area all blocks


}


int main(){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	printf("m %ld , n% ld\n",m, n);
	
	/*int out[4] = {};	*/

	/*get_right_bottom_rectangle(8, 10, m, n, data, out);*/
	/*printf("--->x1 %d    x2 %d   y1 %d   y2 %d   \n", out[0],  out[1],  out[2],  out[3]);*/
	/*get_left_bottom_rectangle(8, 10, m, n, data, out);*/
	/*printf("--->x1 %d    x2 %d   y1 %d   y2 %d   \n", out[0],  out[1],  out[2],  out[3]);*/
	/*get_left_top_rectangle(8, 10, n, data, out);*/
	/*printf("--->x1 %d    x2 %d   y1 %d   y2 %d   \n", out[0],  out[1],  out[2],  out[3]);*/
	/*get_right_top_rectangle(8, 10, n, data, out);*/
	/*printf("--->x1 %d    x2 %d   y1 %d   y2 %d   \n", out[0],  out[1],  out[2],  out[3]);*/

	printf("\n\n");

	// CUDA
	//    number of tests = grid_x*grid_y	
	int grid_x = 2;
	int grid_y = 2;
	
	int *data_d;
	int *areas;

	// Get Mem
	cudaMalloc((void **)&data_d, sizeof(int)*m*n );
	cudaMalloc((void **)&areas, sizeof(int)*grid_x*grid_y ); 

	// Copy data to device memory
	cudaMemcpy(data_d, data, sizeof(int)*m*n, cudaMemcpyHostToDevice);

	dim3 grid(grid_x, grid_y, 1);
	dim3 block(1, 4, 1); // fixed size
	
	int idx_i = 10;
	int idx_j = 15;	

	find_largest_rectangle<<<grid, block>>>(idx_i, idx_j, m, n, data_d, areas);
	cudaDeviceSynchronize();


	return 0;
}
