#  Assignment II: CUDA Basics I

### Exercise 1 - Your first CUDA program and GPU performance metrics

#### 1. Explain how the program is compiled and run. 
First, we need to compile the program(.c) with the NVIDIA CUDA compiler (nvcc) and get the output compiled executable file(.out). Then we need to run the comipled executable file directly. 

#### 2. For a vector length of N:

1. How many floating operations are being performed in your vector add kernel?
  
2. How many global memory reads are being performed by your kernel?
   
#### 3. For a vector length of 1024:

1. Explain how many CUDA threads and thread blocks you used.

2 .Profile your program with Nvidia Nsight. What Achieved Occupancy did you get? You might find https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-metric-comparison useful.

#### 4 .Now increase the vector length to 131070:

1. Did your program still work? If not, what changes did you make?

2. Explain how many CUDA threads and thread blocks you used.

3. Profile your program with Nvidia Nsight. What Achieved Occupancy do you get now?

#### 5.Further increase the vector length (try 6-10 different vector length), plot a stacked bar chart showing the breakdown of time including (1) data copy from host to device (2) the CUDA kernel (3) data copy from device to host. For this, you will need to add simple CPU timers to your code regions.

### Exercise 2 - 2D Dense Matrix Multiplication

#### 1. Name three applications domains of matrix multiplication.

#### 2. How many floating operations are being performed in your matrix multiply kernel? 

#### 3. How many global memory reads are being performed by your kernel?  

#### 4. For a matrix A of (128x128) and B of (128x128):

1. Explain how many CUDA threads and thread blocks you used.
  
2. Profile your program with Nvidia Nsight. What Achieved Occupancy did you get?

#### 5. For a matrix A of (511x1023) and B of (1023x4094):

1. Did your program still work? If not, what changes did you make?

2. Explain how many CUDA threads and thread blocks you used.

3. Profile your program with Nvidia Nsight. What Achieved Occupancy do you get now?

#### 6. Further increase the size of matrix A and B, plot a stacked bar chart showing the breakdown of time including (1) data copy from host to device (2) the CUDA kernel (3) data copy from device to host. For this, you will need to add simple CPU timers to your code regions. Explain what you observe.

#### 7. Now, change DataType from double to float, re-plot the a stacked bar chart showing the time breakdown. Explain what you observe. 
