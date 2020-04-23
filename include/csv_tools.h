//
// Created by Juan Francisco on 2020-02-05.
//

#ifndef BOOSTED_ARD_IO_TOOLS_H
#define BOOSTED_ARD_IO_TOOLS_H

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <boost/algorithm/string.hpp>


// Reading data in csv
struct csv_data_t{
    std::vector<double> data_vec;
    std::string header;
    long m{};
    long n{};

    double get(int i, int j){
        return data_vec[i*n + j];
    }

    int get_int(int i, int j){
        return static_cast<int>(data_vec[i * n + j]);
    }

    void print_data(){
        std::cout<< "HEADER: " << header << std::endl;
        std::cout << "m " << m << " n " << n << std::endl;
        std::cout << "data: " << std::endl;
        for (auto i=0; i<m; i++){
            for (auto j=0; j<n; j++){
                std::cout << data_vec[i*n + j] << " ";
            }
            std::cout<<std::endl;
        }
    }
};


/**
 * Read CSV. File could include headers but it must be composed only of numerical data stored in double format.
 */
void read_numerical_csv(const std::string &file_name, bool has_header, csv_data_t &csv_data){

    long m = 0;
    long n = 0;
    long counter = 0;

    std::ifstream file(file_name);
    std::string str;

    // Getting first line
    std::getline(file, str);
    std::vector<std::string> vec_first;
    boost::algorithm::split(vec_first, str, boost::is_any_of(","));

    n = vec_first.size();
    csv_data.n = n;

    if (!has_header){
        for(const std::string &data : vec_first) {
            csv_data.data_vec.push_back(std::stoi(data));
        }
        counter++;
    }else{
        csv_data.header = str;
    }

    while (std::getline(file, str)){
        std::vector<std::string> vec;
        boost::algorithm::split(vec, str, boost::is_any_of(","));
        if(vec.size() != n){
            throw 0;    // CSV Data are not homogeneous, size of each row is different.
        }
        for(const std::string &data : vec){
            csv_data.data_vec.push_back(std::stoi( data ));
        }
        counter++;
    }

    m = counter;
    csv_data.m = m;
}


#endif //BOOSTED_ARD_IO_TOOLS_H
