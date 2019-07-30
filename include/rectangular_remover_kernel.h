
// GPU kernels
__global__ void remove_rectangle_from_matrix(int x1, int x2, int y1, int y2, int *data_matrix, long m, long n){

	int i = threadIdx.y;
	int j = threadIdx.x;
	
	int g_i = blockDim.y*blockIdx.y + i;
	int g_j = blockDim.x*blockIdx.x + j;
	
	int g_ii = g_i + y1;
	int g_jj = g_j + x1;

	if ( (g_ii >= y1) & (g_ii <= y2)){
		if ( (g_jj >= x1) & (g_jj <= x2)){
			data_matrix[g_ii*n + g_jj] = 0;		
		}
	}
}


