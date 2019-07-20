
#include <math.h>
// getters
#include "./getters.h"
// random
#include "./random_generator.h"

 /* 
    FIND THE LARGEST RECTANGLE:
    Kernel used for expand a rectangle as large as possible in the data matrix
    The kernel runs in different blocks, each block with 4 threads.

    Each block needs a random point in the matrix(it uses cuRand). 
    If it doesnt find a valid point, the block sleeps(note: other blocks can work if they find a valid point).
    
    Then, inside of each block the four threads expand the rectangle from the initial random point to all sides:
    [thread 0] right-bottom
    [thread 1] right-top
    [thread 2] left-bottom
    [thread 3] left-top

    Each block has a largest rectangle that could find, in order to get the largest Rectangle, a reduction is performed
    using CUDA AtomicMax.

    The reduction process works on a grid of size(n,4), where n=0,1,2,3....
    The number of tests goes from [4 to n*4], max number of test available is: 600*4

    The reduction starts at row level, it means, getting the max area of: block[0,:], block[1,:], ... , block[n,:]
    Lastly the reduction is performed at grid level. The rectangle with the max area is used and store in (int *out)

 */
__global__ void find_largest_rectangle(curandState *state, long m, long n, int *data_matrix, int *out, int *areas){


	const int coords_m = 5;
	const int coords_n = 4;

	__shared__ int coords[coords_m * coords_n];
	//__shared__ int total_max;

	__shared__ int idx_i;
	__shared__ int idx_j;
	
	__shared__ bool is_sleeping;

	//int i = threadIdx.y;
	int j = threadIdx.x;

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = gridDim.x;
	
	
	/* GET RANDOM POINT: the value of that random point in the matrix must be one(1)
	 */
	if(j==0){
	        areas[b_i*b_n + b_j] = 0;

		int id = b_i*b_n + b_j;
		curandState localState = state[id];
		
		unsigned int xx;
		unsigned int yy;
		for(int g=0; g<100; g++){
			xx = curand(&localState);
			yy = curand(&localState);
		 	idx_i = abs((int)xx)%m;	
			idx_j = abs((int)yy)%n;
			if (data_matrix[idx_i*n + idx_j]==1){
				is_sleeping = false;
				break;
			}else{
				is_sleeping = true;
			}
		}
		state[id] = localState;

	}
	__syncthreads();
	
	// if sleeping true ,disable block-thread work
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
			out[b_i*b_n*4 + (4*b_j + j) ] = pl;
		}

		if (j==1){
			int a = coords[0*coords_n + 1];
			int b = coords[1*coords_n + 1];
			int pr = a;
			if (b < a){
				pr = b;
			}
			coords[4*coords_n + j] = pr;
			out[b_i*b_n*4 + (4*b_j + j) ] = pr;
		}

		if (j==2){
			int a = coords[1*coords_n + 3];
			int b = coords[3*coords_n + 3];
			int pt = a;
			if (b > a){
				pt = b;
			}
			coords[4*coords_n + j] = pt;
			out[b_i*b_n*4 + (4*b_j + j) ] = pt;
		}

		if (j==3){
			int a = coords[0*coords_n + 3];
			int b = coords[2*coords_n + 3];
			int pb = a;
			if (b < a){
				pb = b;
			}
			coords[4*coords_n + j] = pb;
			out[b_i*b_n*4 + (4*b_j + j) ] = pb;
		}

		__syncthreads();

		if (j==0){
			int a = abs(coords[coords_n*4 + 0] -  coords[coords_n*4 + 1]) + 1;
			int b = abs(coords[coords_n*4 + 2] -  coords[coords_n*4 + 3]) + 1;
			int area = a*b;
			areas[b_i*b_n + b_j] = area;
		}
	}

}

