
#include "stdio.h"
#include "stdlib.h"

//data
#include "./data/squares.h"



void get_right_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg);
void get_left_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg);
void get_left_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg);
void get_right_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg);

int main(){
	printf("adaptive-boxes-gpu\n");
	printf("GPU-accelerated rectangular decomposition for sound propagation modeling\n");

	printf("m %ld , n% ld\n",m, n);

	
	get_right_bottom_rectangle(0, 4, m, n, data);
	get_left_bottom_rectangle(0, 4, m, n, data);
	get_left_top_rectangle(0, 4, n, data);
	get_right_top_rectangle(0, 4, n, data);






	return 0;
}


void get_right_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg){

	int step_j = 0;
	int first_step_i = 0;

	int i_val = 0;
	int j_val = 0;
	int temp_val = 0;
	int step_i = 0;
	
	int x1_val = 0;
	int x2_val = 0;
	int y1_val = 0;
	int y2_val = 0;


	while (true){
		i_val = idx_i_arg;
		j_val = idx_j_arg + step_j;

		if(j_val == n_arg){
			break;
		}

		temp_val = data_matrix_arg[i_val * n_arg + j_val];
		if (temp_val == 0){
			break;
		}

		step_i = 0;

		while (true){
			i_val = idx_i_arg + step_i;

			if (i_val == m_arg){
				break;
			}

			temp_val = data_matrix_arg[i_val * n_arg + j_val];

			if(temp_val == 0){
				break;
			}
			step_i++;
		}


		if (step_j == 0){
			first_step_i = step_i;
		}else{
			if(step_i < first_step_i){
				break;
			}
		}
		step_j++;
	}

	x1_val = idx_j_arg;
	y1_val = idx_i_arg;
	x2_val = idx_j_arg + step_j - 1;
	y2_val = idx_i_arg + first_step_i - 1;

	printf("x1 %d   x2 %d    y1 %d    y2 %d\n", x1_val, x2_val, y1_val, y2_val);

}


void get_left_bottom_rectangle(int idx_i_arg, int idx_j_arg, long m_arg, long n_arg, int *data_matrix_arg){

	int step_j = 0;
	int first_step_i = 0;

	int i_val = 0;
	int j_val = 0;
	int temp_val = 0;
	int step_i = 0;
	
	int x1_val = 0;
	int x2_val = 0;
	int y1_val = 0;
	int y2_val = 0;


	while (true){
		i_val = idx_i_arg;
		j_val = idx_j_arg - step_j;

		if(j_val == -1){
			break;
		}

		temp_val = data_matrix_arg[i_val * n_arg + j_val];
		if (temp_val == 0){
			break;
		}

		step_i = 0;
		while (true){
			i_val = idx_i_arg + step_i;

			if (i_val == m_arg){
				break;
			}

			temp_val = data_matrix_arg[i_val * n_arg + j_val];

			if(temp_val == 0){
				break;
			}
			step_i++;
		}


		if (step_j == 0){
			first_step_i = step_i;
		}else{
			if(step_i < first_step_i){
				break;
			}
		}
		step_j++;
	}

	x1_val = idx_j_arg;
	y1_val = idx_i_arg;
	x2_val = idx_j_arg - step_j + 1;
	y2_val = idx_i_arg + first_step_i - 1;

	printf("x1 %d   x2 %d    y1 %d    y2 %d\n", x1_val, x2_val, y1_val, y2_val);

}


void get_left_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg){

	int step_j = 0;
	int first_step_i = 0;

	int i_val = 0;
	int j_val = 0;
	int temp_val = 0;
	int step_i = 0;
	
	int x1_val = 0;
	int x2_val = 0;
	int y1_val = 0;
	int y2_val = 0;


	while (true){
		i_val = idx_i_arg;
		j_val = idx_j_arg - step_j;

		if(j_val == -1){
			break;
		}

		temp_val = data_matrix_arg[i_val * n_arg + j_val];
		if (temp_val == 0){
			break;
		}

		step_i = 0;
		while (true){
			i_val = idx_i_arg - step_i;

			if (i_val == -1){
				break;
			}

			temp_val = data_matrix_arg[i_val * n_arg + j_val];

			if(temp_val == 0){
				break;
			}
			step_i++;
		}


		if (step_j == 0){
			first_step_i = step_i;
		}else{
			if(step_i < first_step_i){
				break;
			}
		}
		step_j++;
	}

	x1_val = idx_j_arg;
	y1_val = idx_i_arg;
	x2_val = idx_j_arg - step_j + 1;
	y2_val = idx_i_arg - first_step_i + 1;

	printf("x1 %d   x2 %d    y1 %d    y2 %d\n", x1_val, x2_val, y1_val, y2_val);

}


void get_right_top_rectangle(int idx_i_arg, int idx_j_arg, long n_arg, int *data_matrix_arg){

	int step_j = 0;
	int first_step_i = 0;

	int i_val = 0;
	int j_val = 0;
	int temp_val = 0;
	int step_i = 0;
	
	int x1_val = 0;
	int x2_val = 0;
	int y1_val = 0;
	int y2_val = 0;


	while (true){
		i_val = idx_i_arg;
		j_val = idx_j_arg + step_j;

		if(j_val == n_arg){
			break;
		}

		temp_val = data_matrix_arg[i_val * n_arg + j_val];
		if (temp_val == 0){
			break;
		}

		step_i = 0;

		while (true){
			i_val = idx_i_arg - step_i;

			if (i_val == -1){
				break;
			}

			temp_val = data_matrix_arg[i_val * n_arg + j_val];

			if(temp_val == 0){
				break;
			}
			step_i++;
		}


		if (step_j == 0){
			first_step_i = step_i;
		}else{
			if(step_i < first_step_i){
				break;
			}
		}
		step_j++;
	}

	x1_val = idx_j_arg;
	y1_val = idx_i_arg;
	x2_val = idx_j_arg + step_j - 1;
	y2_val = idx_i_arg - first_step_i + 1;

	printf("x1 %d   x2 %d    y1 %d    y2 %d\n", x1_val, x2_val, y1_val, y2_val);

}

