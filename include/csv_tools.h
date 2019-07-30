

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include "./rectangle.h"

// Reading data in csv format
//

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

