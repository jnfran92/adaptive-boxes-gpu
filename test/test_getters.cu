
#include <cuda.h>
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
// getters
#include "../include/getters.h"
#include "../include/getters_improved.h"
// data
#include "../data/squares.h"


int main(int argc, char *argv[]){
	printf("adaptive-boxes-gpu\n");
	printf("Test getters\n");

	printf("----> Data size: m %ld , n% ld\n",m, n);


	int out[4] = {};

	int point_to_test_i;
	int point_to_test_j;

	// Test
	point_to_test_i = 5;
	point_to_test_j = 5;
	
	if (data[point_to_test_i*n + point_to_test_j] == 1){
		printf("is valid point\n");
	}

	// old getters
	og::get_right_bottom_rectangle(
			point_to_test_i,
			point_to_test_j,
			m,
			n,
			data,
			out
			);
	printf("--->x1 %d    x2 %d   y1 %d   y2 %d   \n", out[0],  out[1],  out[2],  out[3]);

	// new getters
	ng::get_right_bottom_rectangle(
			point_to_test_i,
			point_to_test_j,
			m,
			n,
			data,
			out
			);
	printf("--->x1 %d    x2 %d   y1 %d   y2 %d   \n", out[0],  out[1],  out[2],  out[3]);



	return 0;
}
