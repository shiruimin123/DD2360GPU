#  Assignment IV: Advanced CUDA

### Exercise 1 - Thread Scheduling and Execution Efficiency 
#### 1. Assume X=800 and Y=600. Assume that we decided to use a grid of 16X16 blocks. That is, each block is organized as a 2D 16X16 array of threads. How many warps will be generated during the execution of the kernel? How many warps will have control divergence? Please explain your answers.

#### 2. Now assume X=600 and Y=800 instead, how many warps will have control divergence? Please explain your answers.

#### 3. Now assume X=600 and Y=799, how many warps will have control divergence? Please explain your answers.

### Exercise 2 - CUDA Streams
#### 1. Compared to the non-streamed vector addition, what performance gain do you get? Present in a plot ( you may include comparison at different vector length)


#### 2. Use nvprof to collect traces and the NVIDIA Visual Profiler (nvvp) to visualize the overlap of communication and computation. To use nvvp, you can check Tutorial: NVVP - Visualize nvprof Traces

We use command```nvprof --output-profile lab4exercise1.nvvp -f ./lab4exercise1 262144``` to trace the performance and use nvvp to check the visualized file.

The vector size is set to 262144. From the figure we can see that the overlap of copying data from host to device and copying data from device to host, launching the kernel for computing.

![The overlap of communication and computation](./images/ex2q2.png)
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
In the experiment, we fixed dimX at 128 and continued to increase the number of iterations. From the figure, we can find that the relative error decreases exponentially as the number of iterations increases.

#### 3. Compare the performance with and without the prefetching in Unified Memory. How is the performance impact? 
We design an input as the flag bit FLAG in the code and perform an AND operation with the if condition of prefetch, so that we can control the prefetch operation without modifying the code.

In the experiment, we tested the floating-point number operation volume and the execution time of the iteration at the same time. We noticed that the floating-point number operation volume when using prefetching operation was improved to a certain extent compared with the floating-point number operation volume when not using prefetching operation.

When observing the execution time of iterations, we found that using prefetch operations seems to reduce the execution time of iterations. At the same time, the time consumption of prefetch operations is several orders of magnitude less than the execution time. However, it is worth mentioning that when observing the execution time of the two, we found that there is very large fluctuation in the execution time, and the observation results about the execution time need further verification.
