
#include "stdio.h"
#include "stdlib.h"
#include <cooperative_groups.h>

// getters
#include "./include/getters.h"

//data
#include "./data/complex.h"


// GPU kernel
__global__ void find_largest_rectangle(int idx_i, int idx_j, long m, long n, int *data_matrix, int* areas){

	using namespace cooperative_groups;

	const int coords_m = 5;
	const int coords_n = 4;

	__shared__ int coords[coords_m * coords_n];
	__shared__ int total_max;


	/*int idx_i = *((int *)args[0]);*/
	/*int idx_j = *((int *)args[1]);*/
	/*long m = *((long *)args[2]);*/
	/*long n = *((long *)args[3]);*/
	/*int *data_matrix = ((int *)args[4]);*/
	/*int *areas = ((int *)args[5]);*/

	/*printf("idx_i %d, idx_j %d\n",idx_i,idx_j);*/


	int i = threadIdx.y;
	int j = threadIdx.x;


	/*int g_i = blockDim.y * blockIdx.y + i;*/
	/*int g_j = blockDim.x * blockIdx.x + j;*/

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = gridDim.x;


	/*printf("i %d  j %d   gi %d   gj %d \n",i,j,g_i,g_j);*/
	
	int results[4] = {};

	if (j==0){
		get_right_bottom_rectangle(idx_i + b_i, idx_j+ b_j, m, n, data_matrix, results);
	}

	if (j==1){
		get_right_top_rectangle(idx_i + b_i, idx_j+ b_j, n, data_matrix, results);
	}

	if (j==2){
		get_left_bottom_rectangle(idx_i + b_i, idx_j+ b_j, m, n, data_matrix, results);
	}

	if (j==3){
		get_left_top_rectangle(idx_i + b_i, idx_j+ b_j, n, data_matrix, results);
	}

	/*printf("x1 %d    x2 %d   y1 %d   y2 %d   \n", results[0],  results[1],  results[2],  results[3]);*/
	coords[j*coords_n + 0] = results[0];
	coords[j*coords_n + 1] = results[1];
	coords[j*coords_n + 2] = results[2];
	coords[j*coords_n + 3] = results[3];

	__syncthreads();


	if (j==0){
		int a = coords[2*coords_n + 1];
		int b = coords[3*coords_n + 1];
		int pl = a;
		if (b > a){
			pl = b;
		}
		coords[4*coords_n + j] = pl;
	}

	if (j==1){
		int a = coords[0*coords_n + 1];
		int b = coords[1*coords_n + 1];
		int pr = a;
		if (b < a){
			pr = b;
		}
		coords[4*coords_n + j] = pr;
	}

	if (j==2){
		int a = coords[1*coords_n + 3];
		int b = coords[3*coords_n + 3];
		int pt = a;
		if (b > a){
			pt = b;
		}
		coords[4*coords_n + j] = pt;
	}

	if (j==3){
		int a = coords[0*coords_n + 3];
		int b = coords[2*coords_n + 3];
		int pb = a;
		if (b < a){
			pb = b;
		}
		coords[4*coords_n + j] = pb;
	}

	__syncthreads();

	/*printf("final x1 %d,   x2 %d,    y1 %d,    y2 %d \n",coords[4*coords_n + 0], coords[4*coords_n + 1],coords[4*coords_n + 2] ,coords[4*coords_n + 3]);*/

	// get area
	if (j==0){
		int a = abs(coords[coords_n*4 + 0] -  coords[coords_n*4 + 1]);
		int b = abs(coords[coords_n*4 + 2] -  coords[coords_n*4 + 3]);
		int area = a*b;
	 	coords[coords_n*0 + 0] = area; // Saving area in coords[0][0]
	       	areas[b_i*b_n + b_j] = area;
		/*printf("bi %d    bj %d     bn %d\n", b_i,b_j,b_n);*/
		/*printf("a %d    b %d    area %d, area mat %d\n", a,b,area, areas[b_i*b_n + b_j]);*/
		/*printf("area %d\n", area);*/
	}
	__syncthreads();

	// get max area all blocks
	if (b_j == 0){
		/*int temp_area = coords[0];*/
		/*printf("bj %d     bi %d     j %d\n",b_j,b_i, j);*/
		int temp_area = areas[b_i*b_n + j];
		/*printf("temp_area %d    j %d \n", temp_area, j);*/
		atomicMax(&total_max, temp_area);
		__syncthreads();

		if(j == 0){
			printf("total max %d  of block %d\n", total_max, b_i);
			/*areas[b_i*b_n + 0] = total_max;			*/
			/*atomicMax(&areas[0*b_n + 0], total_max);*/
			/*printf("total_max %d -  bi  %d \n", total_max, b_i);*/
		}
	}
	__syncthreads();



	/*__syncthreads();*/

	grid_group grid = this_grid();
	grid.sync();

	// get final x1 x2 y1 y2
	if (j == 0){
		int a = coords[coords_n*0 + 0];
		int b = areas[b_n*0 + 0];
		printf("a %d, b %d\n",a,b);

		if(a==b){
			printf("final x1 %d,    x2 %d,    y1 %d,    y2 %d \n",coords[4*coords_n + 0], coords[4*coords_n + 1],coords[4*coords_n + 2] ,coords[4*coords_n + 3]);
		}
	}
}


__global__ void kernel(int *data, long m){
	using namespace cooperative_groups;
	grid_group g = this_grid();
	printf("it works %ld!!\n",m);
}

int main(){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	printf("m %ld , n% ld\n",m, n);
	
	/*int out[4] = {};	*/

	/*int a = 1;*/
	/*long b = 2;*/
	/*int c[3] = {1,2,3};*/
	/*void *args[] = {&a, &b, c};*/
	/*print_pointers(args);*/
	
	


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
	int grid_x = 4; // fixedint grid_y = 50; //
	int grid_y = 10; //
	
	int *data_d;
	int *areas_d;

	// Get Mem
	cudaMalloc((void **)&data_d, sizeof(int)*m*n );
	cudaMalloc((void **)&areas_d, sizeof(int)*grid_x*grid_y ); 

	// CPU memKE

	int *areas = new int[grid_x*grid_y];

	// Copy data to device memory
	cudaMemcpy(data_d, data, sizeof(int)*m*n, cudaMemcpyHostToDevice);

	dim3 grid(grid_x, grid_y, 1);
	dim3 block(4, 1, 1); // fixed size
	
	int idx_i = 100;
	int idx_j = 100;	


	void *kernel_args[] = {&idx_i, &idx_j, &m, &n, &data_d, &areas_d};
	
	/*find_largest_rectangle_params<<<grid, block>>>(params_ptr);*/
	/*find_largest_rectangle<<<grid, block>>>(idx_i, idx_j, m, n, data_d, areas_d);*/
	
	cudaLaunchCooperativeKernel((void *)find_largest_rectangle, grid, block, kernel_args);
	cudaDeviceSynchronize();
	
	cudaMemcpy(areas, areas_d, sizeof(int)*grid_x*grid_y, cudaMemcpyDeviceToHost);

	printf("areas %d \n ", areas[0]);


	return 0;
}
