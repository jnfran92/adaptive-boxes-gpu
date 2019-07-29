
#include <cuda.h>
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
// getters
#include "../include/getters.h"
#include "../include/getters_improved.h"
// data
#include "../data/hall10.h"


bool verify_results(int *expected, int *actual, bool print);

int main(int argc, char *argv[]){
	printf("adaptive-boxes-gpu\n");
	printf("Test getters\n");

	printf("----> Data size: m %ld , n% ld\n",m, n);


	int out_expected[4] = {};
	int out[4] = {};

	int point_to_test_i;
	int point_to_test_j;

	// Test
	for (int idx_j=0; idx_j<n; idx_j++){
		
		for (int idx_i=0; idx_i<m; idx_i++){
		
			point_to_test_i = 0 + idx_i;
			point_to_test_j = 0 + idx_j;
			
			if (data[point_to_test_i*n + point_to_test_j] == 1){
				printf("Testing in i %d j %d ---------------\n",
						point_to_test_i
						,point_to_test_j);


				bool is_ok;
				
				printf("Test RB\n");
				// old getters
				og::get_right_bottom_rectangle(
						point_to_test_i,
						point_to_test_j,
						m,
						n,
						data,
						out_expected
						);

				// new getters
				ng::get_right_bottom_rectangle(
						point_to_test_i,
						point_to_test_j,
						m,
						n,
						data,
						out
						);
				
				is_ok = verify_results(out_expected, out, true);
				printf("\n");

				
				printf("Test LB\n");
				// old getters
				og::get_left_bottom_rectangle(
						point_to_test_i,
						point_to_test_j,
						m,
						n,
						data,
						out_expected
						);

				// new getters
				ng::get_left_bottom_rectangle(
						point_to_test_i,
						point_to_test_j,
						m,
						n,
						data,
						out
						);
				
				is_ok = verify_results(out_expected, out, true);
				printf("\n");


				printf("Test LT\n");
				// old getters
				og::get_left_top_rectangle(
						point_to_test_i,
						point_to_test_j,
						n,
						data,
						out_expected
						);

				// new getters
				ng::get_left_top_rectangle(
						point_to_test_i,
						point_to_test_j,
						n,
						data,
						out
						);
				
				is_ok = verify_results(out_expected, out, true);
				printf("\n");

				
				printf("Test RT\n");
				// old getters
				og::get_right_top_rectangle(
						point_to_test_i,
						point_to_test_j,
						n,
						data,
						out_expected
						);

				// new getters
				ng::get_right_top_rectangle(
						point_to_test_i,
						point_to_test_j,
						n,
						data,
						out
						);
				
				is_ok = verify_results(out_expected, out, true);
				printf("\n");
			}
		}
	}

	return 0;
}


bool verify_results(int *expected, int *actual, bool print){



	bool test_passed = false;
	if ( (expected[0] != actual[0]) ||
		        (expected[1] != actual[1]) ||
 			(expected[2] != actual[2]) ||
 			(expected[3] != actual[3]) ){
		if (print)
		printf("Test Failed\n");

		if (print)
		printf("EXPECTED: x1 %d    x2 %d   y1 %d   y2 %d   \n",
				expected[0],
				expected[1], 
				expected[2], 
				expected[3]);


		if (print)
		printf("ACTUAL: x1 %d    x2 %d   y1 %d   y2 %d   \n", 
				actual[0],  
				actual[1],  
				actual[2],  
				actual[3]);

	}else{
		if (print)
		printf("Test Passed\n");
		test_passed = true;
	}

	return test_passed;
}




