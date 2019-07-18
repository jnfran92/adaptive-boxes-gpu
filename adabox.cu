
#include <curand_kernel.h>
#include <cuda.h>

#include "stdlib.h"
#include <math.h>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <string>
// thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
// getters
#include "./include/getters.h"
// random
#include "./include/random_generator.h"
//data
#include "./data/boston12.h"
//STL
#include <vector>

#define CC(x) do { if((x) != cudaSuccess) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__); \
	    return EXIT_FAILURE;}} while(0)

// GPU kernels
__global__ void remove_rectangle_from_matrix(int *coords, int *data_matrix, long m, long n){

	int i = threadIdx.y;
	int j = threadIdx.x;
	
	int g_i = blockDim.y*blockIdx.y + i;
	int g_j = blockDim.x*blockIdx.x + j;

	int x1 = coords[0];
	int x2 = coords[1];
	int y1 = coords[2];
	int y2 = coords[3];

	/*printf("gi gj  %d %d\n", g_i, g_j);*/

	if ( (g_i >= y1) & (g_i <= y2)){
		if ( (g_j >= x1) & (g_j <= x2)){
			data_matrix[g_i*n + g_j] = 0;		
		}
	}
}



__global__ void generate_kernel(curandState *state){

	int i = threadIdx.y;
	int j = threadIdx.x;

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = gridDim.x;
	
	if (j==0){
		int id = b_i*b_n + b_j;
		curandState localState = state[id];
		
		unsigned int x;
		for(int g=0; g<100; g++){
		 	x = curand(&localState)%1000;	
			printf(" [%d] %u ",g,x);
		}

		printf("\n\n");
		state[id] = localState;
	}
}


__global__ void find_largest_rectangle(curandState *state, long m, long n, int *data_matrix, int* areas, int *out){

	using namespace cooperative_groups;

	const int coords_m = 5;
	const int coords_n = 4;

	__shared__ int coords[coords_m * coords_n];
	__shared__ int total_max;

	__shared__ int idx_i;
	__shared__ int idx_j;
	
	__shared__ bool is_sleeping;

	int i = threadIdx.y;
	int j = threadIdx.x;

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = gridDim.x;
	
	
	// get random point, must be one on data matrix and set zero all areas values
	if(j==0){

	       	areas[b_i*b_n + b_j] = 0;
		total_max = 0;

		int id = b_i*b_n + b_j;
		curandState localState = state[id];
		
		unsigned int xx;
		unsigned int yy;
		for(int g=0; g<200; g++){
			xx = curand(&localState);
			yy = curand(&localState);
		 	idx_i = abs((int)xx)%m;	
			idx_j = abs((int)yy)%n;
			if (data_matrix[idx_i*n + idx_j]==1){
				/*printf(" found idx_i %d idx_j %d\n ",idx_i, idx_j);*/
				is_sleeping = false;
				break;
			}else{
				is_sleeping = true;
			}
		}
		state[id] = localState;
	}
	__syncthreads();
	
	// if sleeping true disable thread work
	if (!is_sleeping){
	
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

	coords[j*coords_n + 0] = results[0];
	coords[j*coords_n + 1] = results[1];
	coords[j*coords_n + 2] = results[2];
	coords[j*coords_n + 3] = results[3];

	__syncthreads();

	// merge last rectangles
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

	// get area
	if (j==0){
		int a = abs(coords[coords_n*4 + 0] -  coords[coords_n*4 + 1]);
		int b = abs(coords[coords_n*4 + 2] -  coords[coords_n*4 + 3]);
		int area = a*b;
	 	coords[coords_n*0 + 0] = area; // Saving area in coords[0][0]
	       	areas[b_i*b_n + b_j] = area;
		/*printf("area %d\n", area);*/
	}
	__syncthreads();

	// get max area all blocks
	if (b_j == 0){
		int temp_area = areas[b_i*b_n + j];
		/*printf("temp_area %d    j %d \n", temp_area, j);*/
		atomicMax(&total_max, temp_area);
		__syncthreads();

		if(j == 0){
			areas[b_i*b_n + 0] = total_max;			
			atomicMax(&areas[0*b_n + 0], total_max);
			/*printf("total_max %d -  of block  %d \n", total_max, b_i);*/
		}
	}

	}

	grid_group grid = this_grid();
	grid.sync();

	if (!is_sleeping){
	// get final x1 x2 y1 y2
	if (j == 0){
		int a = coords[coords_n*0 + 0];
		int b = areas[b_n*0 + 0];
		/*printf("a %d, b %d\n",a,b);*/

		if(a==b){
			out[0] = coords[4*coords_n + 0];
			out[1] = coords[4*coords_n + 1];
			out[2] = coords[4*coords_n + 2];
			out[3] = coords[4*coords_n + 3];
			/*printf("final x1 %d,    x2 %d,    y1 %d,    y2 %d \n",coords[4*coords_n + 0], coords[4*coords_n + 1],coords[4*coords_n + 2] ,coords[4*coords_n + 3]);*/
		}
	}
	}
}


__global__ void kernel(int *data, long m){
	using namespace cooperative_groups;
	grid_group g = this_grid();
	printf("it works %ld!!\n",m);
}

struct rectangle_t{
	int x1;
	int x2;
	int y1;
	int y2;
};


int main(int argc, char *argv[]){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	printf("m %ld , n% ld\n",m, n);
	printf("\n\n");

	// Rectangles vector
	std::vector<rectangle_t> recs;

	// CUDA
	//    number of tests = grid_x*grid_y	
	std::cout << "grid config" << std::endl;
	int grid_x = 4; // fixed
	int grid_y = atoi(argv[1]); //
	printf("number of tests: %d \n",grid_x*grid_y);

	// GPU data
	int *data_d;
	int *areas_d;
	int *out_d;

	// Thrust Data
	std::cout << "thrust" << std::endl;
	thrust::device_vector<int> t_data_d(m*n);	
	data_d = thrust::raw_pointer_cast(&t_data_d[0]);

	std::cout << "thrust" << std::endl;
	thrust::device_vector<int> t_areas_d(grid_x*grid_y);
	areas_d = thrust::raw_pointer_cast(&t_areas_d[0]);
	
	// Get Mem
	cudaMalloc((void **)&out_d, sizeof(int)*4);

	// CPU mem
	int *areas = new int[grid_x*grid_y];
	int *out = new int[4];

	// Copy data to device memory
	cudaMemcpy(data_d, data, sizeof(int)*m*n, cudaMemcpyHostToDevice);

	// Grid and Block size
	dim3 grid(grid_x, grid_y, 1);
	dim3 block(4, 1, 1); // fixed size
	
	dim3 image_grid(n/2,m/2,1);
	dim3 image_block(2,2,1);


	// curand
	curandState *devStates;
	CC(cudaMalloc((void **)&devStates, grid_x*grid_y*sizeof(unsigned int)));
	
	// args ptr
	void *kernel_args[] = {&devStates, &m, &n, &data_d, &areas_d, &out_d};
	
	// Init algorithm
	// Setup
	setup_kernel<<<grid, block>>>(devStates);
	cudaDeviceSynchronize();
	
	// Loop
	rectangle_t rec;
	int max_step = 2000;
	int sum;
	// init last sum
	int last_sum = thrust::reduce(t_data_d.begin(), t_data_d.end());

	for (int step=0; step<max_step; step++){

		/*thrust::fill(t_areas_d.begin(), t_areas_d.end(), 0);*/
		/*cudaDeviceSynchronize();*/
		
		/*printf("sum %d\n",sum);*/
		/*std::cout << "step" << std::endl;*/
		
		cudaLaunchCooperativeKernel((void *)find_largest_rectangle, grid, block, kernel_args);
		cudaDeviceSynchronize();

		/*sum = thrust::reduce(t_data_d.begin(), t_data_d.end());*/
		/*printf("sum %d\n",sum);*/

		/*std::cout << "step" << std::endl;*/
		remove_rectangle_from_matrix<<<image_grid, image_block>>>(out_d, data_d, m, n);
		cudaDeviceSynchronize();

		/*std::cout << "step" << std::endl;*/
		sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
		printf("sum %d\n",sum);
		cudaDeviceSynchronize();
		
		CC( cudaMemcpy(out, out_d, sizeof(int)*4, cudaMemcpyDeviceToHost)  );
		cudaDeviceSynchronize();

		if(sum < last_sum){
			rec.x1 = out[0];
			rec.x2 = out[1];
			rec.y1 = out[2];
			rec.y2 = out[3];
			recs.push_back(rec);
		}
		last_sum = sum;
		if(sum<=0){
			break;
		}
	}

	cudaMemcpy(data, data_d, sizeof(int)*m*n, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	

	// Saving data in csv format

	std::ofstream r_file;
	std::string file_name = "boston12_";
	file_name += std::to_string(grid_x*grid_y);
	file_name += ".csv";
	r_file.open(file_name);

	std::cout << "saving rectagles -  vector size "<< recs.size() << std::endl;
	std::vector<rectangle_t>::iterator v = recs.begin();
	while(v !=recs.end()){
		/*std::cout <<"  "<< v->x1 <<"  "<< v->x2 <<"  "<< v->y1 <<"  "<< v->y2 << std::endl;*/
		r_file << v->x1 <<",  "<< v->x2 <<",  "<< v->y1 <<",  "<< v->y2 << "\n";
		v++;
	}
	
	r_file.close();



	/*printf("\n\n");*/

	/*for (int i=0; i<m; i++){*/
		/*for (int j=0; j<n; j++){*/
			/*printf("%d ", data[i*n + j]);*/
		/*}*/
		/*printf("\n");*/
	/*}*/

	delete areas;
	delete out;

	cudaFree(devStates);
	cudaFree(out_d);





	return 0;
}
