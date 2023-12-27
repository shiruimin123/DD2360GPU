#  Assignment IV: Advanced CUDA

### Exercise 1 - Thread Scheduling and Execution Efficiency 
#### 1. Assume X=800 and Y=600. Assume that we decided to use a grid of 16X16 blocks. That is, each block is organized as a 2D 16X16 array of threads. How many warps will be generated during the execution of the kernel? How many warps will have control divergence? Please explain your answers.

#### 2. Now assume X=600 and Y=800 instead, how many warps will have control divergence? Please explain your answers.

#### 3. Now assume X=600 and Y=799, how many warps will have control divergence? Please explain your answers.

### Exercise 2 - CUDA Streams
#### 1. Divide an input vector into multiple segments of a given size (S_seg)
#### 2. Create 4 CUDA streams to copy asynchronously from host to GPU memory, perform vector addition on GPU, and copy back the results from GPU memory to host memory
#### 3. What is the impact of segment size on performance? Present in a plot ( you may choose a large vector and compare 4-8 different segment sizes)

### Exercise 3 - Heat Equation with using NVIDIA libraries
#### 1. Run the program with different dimX values. For each one, approximate the FLOPS (floating-point operation per second) achieved in computing the SMPV (sparse matrix multiplication). Report FLOPS at different input sizes in a FLOPS. What do you see compared to the peak throughput you report in Lab2?

We noticed that there are three operations in each iteration: cusparseSpMV, cublasDaxpy and cublasDnrm2 operations. These operations all contain a known number of floating point operations, so we add a FLOPS counter and timer to the code and calculate each second. Floating-point operations: FLOPS = Total Floating-Point Operations / Time.

In the cusparseSpMV operation, assuming that each element in A and temp participates in multiplication and addition operations, there are 2*nzv floating point operations per iteration.

In the cublasDaxpy operation, there are 2*dimX floating point operations per iteration.

In the cublasDnrm2 operation, there are 2*dimX floating point operations per iteration.

We fixed nstep and continuously increased the value of dimX, and obtained the changing pattern of floating-point operations per second as shown in the figure below:

We noticed that although there are certain fluctuations, the amount of floating point operations is basically linearly related to the increase in input data. We were not able to observe peak throughput in this experiment. According to the information, this may be because the iteration of the algorithm limits its operations to short bursts of activity.

#### 2. Run the program with dimX=128 and vary nsteps from 100 to 10000. Plot the relative error of the approximation at different nstep. What do you observe?

#### 3. Compare the performance with and without the prefetching in Unified Memory. How is the performance impact? [Optional: using nvprof to get metrics on UM]
