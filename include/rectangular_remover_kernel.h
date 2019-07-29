
// GPU kernels
__global__ void remove_rectangle_from_matrix(int x1, int x2, int y1, int y2, int *data_matrix, long m, long n){

	int i = threadIdx.y;
	int j = threadIdx.x;
	
	int g_i = blockDim.y*blockIdx.y + i;
	int g_j = blockDim.x*blockIdx.x + j;

	if ( (g_i >= y1) & (g_i <= y2)){
		if ( (g_j >= x1) & (g_j <= x2)){
			data_matrix[g_i*n + g_j] = 0;		
		}
	}
}


// GPU kernels
__global__ void remove_rectangle_from_matrix_improved(int x1, int x2, int y1, int y2, int *data_matrix, long m, long n){

	int i = threadIdx.y;
	int j = threadIdx.x;
	
	int g_i = blockDim.y*blockIdx.y + i;
	int g_j = blockDim.x*blockIdx.x + j;

	//int dist_x = (x2 - x1) + 1;
	//int dist_y = (y2 - y1) + 1;

	int g_ii = g_i + y1;
	int g_jj = g_j + x1;
	//printf("g_ii %d  g_jj %d\n",g_ii, g_jj);

	if ( (g_ii >= y1) & (g_ii <= y2)){
		if ( (g_jj >= x1) & (g_jj <= x2)){
			data_matrix[g_ii*n + g_jj] = 0;		
		}
	}
}


