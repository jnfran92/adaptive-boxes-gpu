all:
	nvcc --std=c++11 -arch=sm_61 -rdc=true adabox.cu -o prog
