# adaptive-boxes-gpu

A GPU-accelerated algorithm for searching an adequate rectangular decomposition of a 2D scene in a reasonable time.


The decomposition algorithm works over a raster image of a scene, which is well represented as a *binary matrix X*.


The algorithm decomposes the *binary matrix X* into large rectangles using `CUDA`.

## Samples

High resolution images! Click over any image to see the rectangular decomposition's details.

### Scene 1



<img src="https://imgur.com/2KAW9Ha.jpg" width="800">



<img src="https://imgur.com/hABkX4i.jpg" width="800">

### Scene 2



<img src="https://imgur.com/LJmxu24.jpg" width="800">



<img src="https://imgur.com/2IsC04x.jpg" width="800">


### Scene 3



<img src="https://imgur.com/dfbjjPf.jpg" width="800">



<img src="https://imgur.com/GHRFJyJ.jpg" width="800">


## Usage Guide

### Requirements
- CUDA 9.0
- Thrust parallel template library
- CuRand

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
A list of resulting rectangles in a `.csv` file. 
Data is given in the format: `[x1 x2 y1 y2]` (Two points rectangle location).

## Performance Test

Execution time in seconds:


| # of parallel searches `[n]` |    Scene 1    |   Scene 2    |   Scene 3    |
|------------------------|--------|--------|--------|
| 2400 |  3.1 |  2.6 |  2 | 

Tests were performed using a GPU NVIDIA Tesla V100.

## Extra info

### How does it work?
See the paper (GPU-accelerated rectangular decomposition for sound propagation modeling in 2D)[https://ieeexplore.ieee.org/document/8966434].

### How to plot the `.csv` results?
Use `adaptive-boxes` python library, see here `[ python implementation link]`.

### Why sound propagation modeling?
See here: `[ python ard link link]`.
