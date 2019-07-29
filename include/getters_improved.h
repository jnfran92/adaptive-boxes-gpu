
#include "stdio.h"
#include "stdlib.h"

// rgn: New  Rectangular Getters
namespace ng{
	// results matrix: [x1 x2 y1 y2]
	__device__ __host__ void get_right_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg, int *results){


		int temp_val = 0;
		
		int x1_val = 0;
		int x2_val = 0;
		int y1_val = 0;
		int y2_val = 0;


		int d0 = 0;
		int i=0;
		int j=0;	
		// Get the max distance in the axis
		for (i=idx_i_arg; i<m_arg; i++){
			temp_val = data_matrix_arg[i * n_arg + idx_j_arg];
			if(temp_val == 0){
				i = i - 1;
				break;
			}	
		}
		d0 = i;

		for (j=idx_j_arg + 1; j<n_arg; j++){
			
			for (i=idx_i_arg; i<=d0; i++){
				temp_val = data_matrix_arg[i * n_arg + j];
				if(temp_val == 0){
					i = i - 1;
					break;
				}	
			}

			if (i < d0){
				j = j -1;
				break;
			}
		}
		
		x1_val = idx_j_arg;
		y1_val = idx_i_arg;
		x2_val =  j;
		y2_val =  d0;

		results[0] = x1_val;
		results[1] = x2_val;
		results[2] = y1_val;
		results[3] = y2_val;

	}


	__device__ __host__ void get_left_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg, int *results){


		int temp_val = 0;
		
		int x1_val = 0;
		int x2_val = 0;
		int y1_val = 0;
		int y2_val = 0;


		int d0 = 0;
		int i=0;
		int j=0;	
		// Get the max distance in the axis
		for (i=idx_i_arg; i<m_arg; i++){
			temp_val = data_matrix_arg[i * n_arg + idx_j_arg];
			if(temp_val == 0){
				i = i - 1;
				break;
			}	
		}
		d0 = i;

		for (j=idx_j_arg - 1; j>=0; j--){

			for (i=idx_i_arg; i<=d0; i++){
				temp_val = data_matrix_arg[i * n_arg + j];
				if(temp_val == 0){
					i = i -1;
					break;
				}	
			}

			if (i < d0){
				j = j + 1;
				break;
			}
		}
		
		x1_val = idx_j_arg;
		y1_val = idx_i_arg;
		x2_val =  j;
		y2_val =  d0;

		
		results[0] = x1_val;
		results[1] = x2_val;
		results[2] = y1_val;
		results[3] = y2_val;
	}


	__device__ __host__ void get_left_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg, int *results){


		int temp_val = 0;
		
		int x1_val = 0;
		int x2_val = 0;
		int y1_val = 0;
		int y2_val = 0;


		int d0 = 0;
		int i=0;
		int j = 0;	
		// Get the max distance in the axis
		for (i=idx_i_arg; i>=0; i--){
			temp_val = data_matrix_arg[i * n_arg + idx_j_arg];
			if(temp_val == 0){
				i = i + 1;
				break;
			}	
		}
		d0 = i;

		for (j=idx_j_arg-1; j>=0; j--){

			for (i=idx_i_arg; i>=d0; i--){
				temp_val = data_matrix_arg[i * n_arg + j];
				if(temp_val == 0){
					i = i + 1;
					break;
				}	
			}

			if (i > d0){
				j = j + 1;
				break;
			}
		}
		
		x1_val = idx_j_arg;
		y1_val = idx_i_arg;
		x2_val =  j;
		y2_val =  d0;


		results[0] = x1_val;
		results[1] = x2_val;
		results[2] = y1_val;
		results[3] = y2_val;
	}


	__device__ __host__ void get_right_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg, int *results){

		int temp_val = 0;
		
		int x1_val = 0;
		int x2_val = 0;
		int y1_val = 0;
		int y2_val = 0;


		int d0 = 0;
		int i=0;
		int j=0;	
		// Get the max distance in the axis
		for (i=idx_i_arg; i>=0; i--){
			temp_val = data_matrix_arg[i * n_arg + idx_j_arg];
			if(temp_val == 0){
				i = i + 1;
				break;
			}	
		}
		d0 = i;

		for (j=idx_j_arg + 1; j<n_arg; j++){
			
			for (i=idx_i_arg; i>=d0; i--){
				temp_val = data_matrix_arg[i * n_arg + j];
				if(temp_val == 0){
					i = i + 1;
					break;
				}	
			}

			if (i > d0){
				j = j - 1;
				break;
			}
		}
		
		x1_val = idx_j_arg;
		y1_val = idx_i_arg;
		x2_val = j;
		y2_val = d0;


		results[0] = x1_val;
		results[1] = x2_val;
		results[2] = y1_val;
		results[3] = y2_val;
	}
}
