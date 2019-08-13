# adaptive-boxes-gpu

A GPU-accelerated algorithm for searching an adequate rectangular decomposition of a 2D scene in a reasonable time.
The decomposition algorithm works over a raster image of a scene, which is well represented as a binary matrix X.
The algorithm decomposes the binary matrix into large rectangles using CUDA.

## Samples

### Scene 1

Model:

<img src="https://imgur.com/2KAW9Ha-.jpg" width="800">

Model Decomposed:

<img src="https://imgur.com/hABkX4i-.jpg" width="800">

### Scene 2

Model:

<img src="https://imgur.com/dfbjjPf-.jpg" width="800">

Model Decomposed:

<img src="https://imgur.com/GHRFJyJ-.jpg" width="800">



## Guide

### Basics
First compile the `adaptive_boxes.cu` script. Just do `make`.

Run `./adabox` with the following arguments: 
- [1] input file (binary matrix in .csv) 
- [2] output file (list of rectangles in .csv) 
- [3] n (# of tests = n x n)

### Input file
The input should be a `.csv` file which contains the matrix size and the binary matrix data. 
Some samples are located in `data` folder. As a simple example see `squares.csv`.

### Output file
A list of resulting rectangles. Data is given in the format: `[x1 x2 y1 y2]` (Two points rectangle location).