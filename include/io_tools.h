#ifndef IO_TOOLS_H
#define IO_TOOLS_H

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include "./rectangle.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>

	
struct binary_matrix_t{
	int *data;
	long m;
	long n;
};

// Reading data in csv
void read_binary_data(std::string file_name, binary_matrix_t *matrix_t){

	std::ifstream file(file_name);
	std::string str;

	long m = 0;
	long n = 0;
	long counter = 0;
	long counter_matrix = 0;
	while (std::getline(file, str)){
		
		if (counter == 0){
			m = std::stoi(str);
			matrix_t->m = m;
			//std::cout << m << std::endl;
		}

		if (counter == 1){
			n = std::stoi(str);
			matrix_t->n = n;
			matrix_t->data = new int[m*n];
			//std::cout << n << std::endl;
		}

		if (counter > 1){
			std::vector<std::string> vec;
			boost::algorithm::split(vec, str, boost::is_any_of(","));
			for(std::string data : vec){
				matrix_t->data[counter_matrix] = std::stoi( data );
				counter_matrix++;
			}
		}
		counter++;
	}
}

// Saving data in csv format
void save_rectangles_in_csv(std::string file_name, std::vector<rectangle_t> *recs){

	std::ofstream r_file;
	r_file.open(file_name);

	std::vector<rectangle_t>::iterator v = recs->begin();
	while(v !=recs->end()){
		r_file << v->x1 <<",  "<< v->x2 <<",  "<< v->y1 <<",  "<< v->y2 << "\n";
		v++;
	}
	r_file.close();
}

#endif
