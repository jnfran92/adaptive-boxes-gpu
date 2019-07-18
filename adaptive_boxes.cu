
#include <cuda.h>
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
// thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
//STL
#include <vector>
// cuda call
#include "./include/cuda_call.h"
// kernels
#include "./include/rectangular_explorer_kernel.h"
#include "./include/rectangular_remover_kernel.h"
// rectangle struct
#include "./include/rectangle.h"
// data
#include "./data/boston12.h"


int main(int argc, char *argv[]){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	printf("----> Data size: m %ld , n% ld\n",m, n);

	// CUDA timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Rectangles vector
	std::vector<rectangle_t> recs;

	// CUDA
	//    number of tests = grid_x*grid_y	
	int grid_x = 4; // fixed
	int grid_y = atoi(argv[1]); //
	printf("----> Number of tests: %d \n",grid_x*grid_y);

	// GPU data
	int *data_d;
	int *areas_d;
	int *out_d;

	// Thrust Data
	thrust::device_vector<int> t_data_d(m*n);	
	data_d = thrust::raw_pointer_cast(&t_data_d[0]);

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
	
	// Init algorithm -----------------------
	// Setup
	cudaEventRecord(start);

	setup_kernel<<<grid, block>>>(devStates);
	cudaDeviceSynchronize();
	
	// Loop
	printf("Working...\n");
	rectangle_t rec;
	int max_step = 8000;
	int sum;
	// init last sum
	int last_sum = thrust::reduce(t_data_d.begin(), t_data_d.end());

	for (int step=0; step<max_step; step++){

		cudaLaunchCooperativeKernel((void *)find_largest_rectangle, grid, block, kernel_args);
		cudaDeviceSynchronize();

		remove_rectangle_from_matrix<<<image_grid, image_block>>>(out_d, data_d, m, n);
		cudaDeviceSynchronize();
		
		sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
		/*printf("sum %d\n",sum);*/
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

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Decomposition ready!!\n");
	printf("-->Elapsed time: %f\n", milliseconds);
	printf("-->Last sum %d\n",sum);
	
	/*cudaMemcpy(data, data_d, sizeof(int)*m*n, cudaMemcpyDeviceToHost);*/
	/*cudaDeviceSynchronize();*/
	

	// Saving data in csv format
	std::ofstream r_file;
	std::string file_name = "./results/boston12_";
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

	// free data
	delete areas;
	delete out;

	cudaFree(devStates);
	cudaFree(out_d);

	return 0;
}
