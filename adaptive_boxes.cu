
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
#include "./data/theatre12.h"


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

	// ratio limit
	double ratio_limit = atof(argv[2]);
	printf("----> Ratio Limit: %f\n", ratio_limit);
	
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
	
	// Get Mem
	/*cudaMalloc((void **)&out_d, sizeof(int)*4*grid_x*grid_y);*/

	// CPU mem
	int *areas = new int[grid_x*grid_y];
	int *out = new int[4*grid_x*grid_y];

	// Copy data to device memory
	cudaMemcpy(data_d, data, sizeof(int)*m*n, cudaMemcpyHostToDevice);
	
	// Grid and Block size
	dim3 grid(grid_x, grid_y, 1);
	dim3 block(4, 1, 1); // fixed size
	
	dim3 image_grid(n/2,m/2,1);
	dim3 image_block(2,2,1);

	
	
	// Init algorithm -----------------------
	cudaEventRecord(start);
	// Setup
	// curand
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
	double ratio = 1.0;
	int last_sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
	int global_sum = last_sum; 
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

			remove_rectangle_from_matrix_improved<<<tmp_grid, tmp_block>>>(x1,x2,y1,y2, data_d, m, n);
			cudaDeviceSynchronize();
			
			sum = thrust::reduce(t_data_d.begin(), t_data_d.end());
			/*cudaDeviceSynchronize();*/
			
			if(sum < last_sum){
				rec.x1 = x1;
				rec.x2 = x2;
				rec.y1 = y1;
				rec.y2 = y2;
				recs.push_back(rec);
			}
			
			 /*metrics*/
			ratio =  (double)sum/(double)global_sum;
			/*printf("sum = %d  ratio = %f\n", sum, ratio);			*/

			last_sum = sum;
			if(sum<=0){
				break;
			}
			last_x1 = x1;
			last_x2 = x2;
			last_y1 = y1;
			last_y2 = y2;	
			
			if (ratio<ratio_limit){
				break;
			}

		}
	}

	/*if (sum!=0){*/
		/*printf("final step\n");*/
		/*cudaMemcpy(data, data_d, sizeof(int)*m*n, cudaMemcpyDeviceToHost);*/
		/*for (int j=0;j<n;j++){*/
			/*for (int i=0;i<m;i++){*/
				/*if(data[i*n + j]==1){*/
					/*rec.x1 = j;*/
					/*rec.x2 = j;*/
					/*rec.y1 = i;*/
					/*rec.y2 = i;*/
					/*recs.push_back(rec);*/
				/*}*/
			/*}*/
		/*}*/
	/*}*/



	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Decomposition ready!!\n");
	printf("-->Elapsed time: %f\n", milliseconds);
	printf("-->Last sum %d\n",sum);
	
	
	// Saving data in csv format
	std::ofstream r_file;
	/*std::string file_name = "./results/";*/
	/*file_name += std::to_string(grid_x*grid_y);*/
	/*file_name += ".csv";*/
	std::string file_name = argv[3];
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

	return 0;
}
