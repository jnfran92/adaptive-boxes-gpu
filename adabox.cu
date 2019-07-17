#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include "stdlib.h"
/*#include "cstdlib"*/
#include <cooperative_groups.h>

// getters
#include "./include/getters.h"

//data
#include "./data/squares.h"


// GPU kernels
__global__ void find_random_numbers(){
	
}

__global__ void remove_rectangle_from_matrix(int *coords, int *data_matrix, int m, int n){

	int i = threadIdx.y;
	int j = threadIdx.x;
	
	int g_i = blockDim.y*blockIdx.y + i;
	int g_j = blockDim.x*blockIdx.x + j;

	int x1 = coords[0];
	int x2 = coords[1];
	int y1 = coords[2];
	int y2 = coords[3];

	if ( (g_i >= y1) & (g_i <= y2)){
		if ( (g_j >= x1) & (g_j <= x2)){
			data_matrix[g_i*n + g_j] = 0;		
		}
	}
}


__global__ void find_largest_rectangle(int idx_ii, int idx_jj, long m, long n, int *data_matrix, int* areas, int *out){

	using namespace cooperative_groups;

	const int coords_m = 5;
	const int coords_n = 4;

	__shared__ int coords[coords_m * coords_n];
	__shared__ int total_max;


	int i = threadIdx.y;
	int j = threadIdx.x;

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = gridDim.x;
	
	
	// get random point, must be one on data matrix
	int idx_i = 0;
	int idx_j = 0;
	if(j==0){
		bool search_flag = false;
		while(!search_flag){
			idx_i = rand() % m;
			idx_j = rand() % n;


			


			if (data_matrix[idx_i*n + idx_j] == 1){
				printf("random idx_i %d   idx_j %d  \n",idx_i, idx_j);
				search_flag = true;
				break;
			}		
		}
	}
	__syncthreads();

	// expand the rectangle
	int results[4] = {0,0,0,0};
	if (j==0){
		get_right_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results);
	}

	if (j==1){
		get_right_top_rectangle(idx_i, idx_j, n, data_matrix, results);
	}

	if (j==2){
		get_left_bottom_rectangle(idx_i, idx_j, m, n, data_matrix, results);
	}

	if (j==3){
		get_left_top_rectangle(idx_i, idx_j, n, data_matrix, results);
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
			/*printf("total max %d  of block %d\n", total_max, b_i);*/
			areas[b_i*b_n + 0] = total_max;			
			atomicMax(&areas[0*b_n + 0], total_max);
			printf("total_max %d -  of block  %d \n", total_max, b_i);
		}
	}
	/*__syncthreads();*/



	/*__syncthreads();*/

	grid_group grid = this_grid();
	grid.sync();

	// get final x1 x2 y1 y2
	if (j == 0){
		int a = coords[coords_n*0 + 0];
		int b = areas[b_n*0 + 0];
		printf("a %d, b %d\n",a,b);

		if(a==b){
			out[0] = coords[4*coords_n + 0];
			out[1] = coords[4*coords_n + 1];
			out[2] = coords[4*coords_n + 2];
			out[3] = coords[4*coords_n + 3];
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
	
	printf("\n\n");

	// CUDA
	//    number of tests = grid_x*grid_y	
	int grid_x = 4; // fixedint grid_y = 50; //
	int grid_y = 1; //
	
	int *data_d;
	int *areas_d;
	int *out_d;

	// Get Mem
	cudaMalloc((void **)&data_d, sizeof(int)*m*n );
	cudaMalloc((void **)&areas_d, sizeof(int)*grid_x*grid_y ); 
	cudaMalloc((void **)&out_d, sizeof(int)*4);
	
	// CPU mem
	int *areas = new int[grid_x*grid_y];
	int *out = new int[4];

	// Copy data to device memory
	cudaMemcpy(data_d, data, sizeof(int)*m*n, cudaMemcpyHostToDevice);

	dim3 grid(grid_x, grid_y, 1);
	dim3 block(4, 1, 1); // fixed size
	
	dim3 image_grid(2,2,1);
	dim3 image_block(n/2,m/2,1);

	int idx_i = 5;
	int idx_j = 5;	


	void *kernel_args[] = {&idx_i, &idx_j, &m, &n, &data_d, &areas_d, &out_d};
	
	/*find_largest_rectangle_params<<<grid, block>>>(params_ptr);*/
	/*find_largest_rectangle<<<grid, block>>>(idx_i, idx_j, m, n, data_d, areas_d);*/

	// Init algorithm

	cudaLaunchCooperativeKernel((void *)find_largest_rectangle, grid, block, kernel_args);
	cudaDeviceSynchronize();

	remove_rectangle_from_matrix<<<image_grid, image_block>>>(out_d, data_d, m, n);
	cudaDeviceSynchronize();

	cudaMemcpy(data, data_d, sizeof(int)*m*n, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	
	printf("----->Result  x1 %d     x2 %d    y1 %d     y2 %d \n ", out[0], out[1], out[2], out[3]);
	printf("\n\n");

	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			printf("%d ", data[i*n + j]);
		}
		printf("\n");
	}


	return 0;
}
