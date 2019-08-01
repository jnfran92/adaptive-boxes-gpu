all:
	nvcc -O3 --std=c++11  adaptive_boxes.cu -o adabox
