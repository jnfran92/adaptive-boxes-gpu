all:
	nvcc --std=c++11 -arch=sm_61 -rdc=true adaptive_boxes.cu -o adabox
