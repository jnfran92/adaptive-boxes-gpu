
#include <stdlib.h>
#include <iostream>
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
// csv
#include "./include/io_tools.h"

int main(int argc, char *argv[]){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	if (argc < 4){
		printf("Error Args: 4 Needed \n[1]input file(binary matrix in .csv)\n[2]output file(list of rectangles in .csv) \n[3]n (# of tests = 4*n)\n");
		return 0;
	}

	// Arguments
	std::string input_file_name = argv[1];
	std::string output_file_name = argv[2];
	int n_tests = atoi(argv[3]);


	// Reading data	
	printf("Reading Data...\n");
	binary_matrix_t data_t;
	read_binary_data(input_file_name, &data_t);
	
	long m = data_t.m;
	long n = data_t.n;
	printf("Data on Memory: Data size: m %ld , n% ld\n",m, n);
	
	// CUDA timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Rectangles vector
	std::vector<rectangle_t> recs;

	// CUDA
	int grid_x = 4; // fixed
	int grid_y = n_tests; //
	printf("Number of tests: %d \n",grid_x*grid_y);
	
	// GPU data
	int *data_d;
	int *areas_d;
	int *out_d;

	// Thrust Data
	thrust::device_vector<int> t_data_d(m*n);	
	data_d = thrust::raw_pointer_cast(&t_data_d[0]);

	thrust::device_vector<int> t_areas_d(grid_x*grid_y);
	areas_d = thrust::raw_pointer_cast(&t_areas_d[0]);

	thrust::device_vector<int> t_out_d(grid_x*grid_y*4);
	out_d = thrust::raw_pointer_cast(&t_out_d[0]);	
	
	// Copy data to device memory
	cudaMemcpy(data_d, data_t.data, sizeof(int)*m*n, cudaMemcpyHostToDevice);
	
	// Grid and Block size
	dim3 grid(grid_x, grid_y, 1);
	dim3 block(4, 1, 1); // fixed size
	
	// Init algorithm -----------------------
	cudaEventRecord(start);
	// Setup cuRand
	curandState *devStates;
	CC(cudaMalloc((void **)&devStates, grid_x*grid_y*sizeof(unsigned int)));
	
	setup_kernel<<<grid, block>>>(devStates);
	cudaDeviceSynchronize();

	// Loop
	printf("Working...\n");
	rectangle_t rec;
	int max_step = 999999;
	int sum;
	
	// init last sum
	int last_sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
	
	int last_x1 = -1;
	int last_x2 = -1;
	int last_y1 = -1;
	int last_y2 = -1;
	
	int x1,x2,y1,y2;

	for (int step=0; step<max_step; step++){
		find_largest_rectangle<<<grid,block>>>(devStates,m,n,data_d,out_d, areas_d);
		cudaDeviceSynchronize();
		
		thrust::device_vector<int>::iterator iter = thrust::max_element(t_areas_d.begin(), t_areas_d.end());
		unsigned int position = iter - t_areas_d.begin();
		int max_val = *iter; 
			
		if (max_val==0){
			continue;
		}

		x1 = t_out_d[position*4 + 0];  
		x2 = t_out_d[position*4 + 1];  
		y1 = t_out_d[position*4 + 2];  
		y2 = t_out_d[position*4 + 3];  


		if (!((last_x1==x1) & (last_x2==x2) & (last_y1==y1) & (last_y2==y2)) ){
			int dist_y = (y2 - y1) + 1;
			int dist_x = (x2 - x1) + 1;
			int x_blocks = (int)ceil((double)dist_x/2.0);
			int y_blocks = (int)ceil((double)dist_y/2.0);
			
			dim3 tmp_block(2, 2, 1);
			dim3 tmp_grid(x_blocks, y_blocks, 1);

			remove_rectangle_from_matrix<<<tmp_grid, tmp_block>>>(x1,x2,y1,y2, data_d, m, n);
			cudaDeviceSynchronize();
			
			sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
			
			if(sum < last_sum){
				rec.x1 = x1;
				rec.x2 = x2;
				rec.y1 = y1;
				rec.y2 = y2;
				recs.push_back(rec);
			}
			
			/*printf("sum = %d\n", sum);			*/

			last_sum = sum;
			if(sum<=0){
				break;
			}
			
			last_x1 = x1;
			last_x2 = x2;
			last_y1 = y1;
			last_y2 = y2;	
		}
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Decomposition ready!!\n");
	printf("-->Elapsed time: %f\n", milliseconds);
	printf("-->Last sum %d\n",sum);
	
	
	/*Saving data in csv format*/
	std::cout << "Saving rectagles -  total amount of rectangles: "<< recs.size() << std::endl;
	save_rectangles_in_csv(output_file_name, &recs);	
	
	// Free memory
	cudaFree(devStates);
	/*delete data;*/

	return 0;
}
