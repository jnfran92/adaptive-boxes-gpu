
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

