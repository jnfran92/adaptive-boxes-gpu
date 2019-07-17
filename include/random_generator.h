
#include <cuda.h>
#include <curand_kernel.h>


__global__ void setup_kernel(curandState *state)
{

	int i = threadIdx.y;
	int j = threadIdx.x;

	int b_i = blockIdx.y;
	int b_j = blockIdx.x;
	int b_n = gridDim.x;

	if (j==0){
		int id = b_i*b_n + b_j;
		curand_init(1234, id, 0, &state[id]);
	}
}








