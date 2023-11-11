# Assignment I:

## Exercise 1 - Reflection on GPU-accelerated Computing

#### 1. List the main differences between GPUs and CPUs in terms of architecture.
##### 1. Parallel:
GPU is highly parallel processors which can handle multiple tasks simultaneously. GPU has large number of cores, each capable of executing its own set of instructions independently.

CPU is optimized for single-threaded performance. CPU has higher clock speeds and more complex cores.
##### 2. Number of Cores:
GPU has larger number of simple cores. This allows them to process many simple tasks simultaneously, making them suitable for parallelizable tasks.

CPUs has fewer but more powerful cores that are optimized for handling complex tasks and executing a variety of instructions.
#### 2.Check the latest Top500 list that ranks the top 500 most powerful supercomputers in the world. In the top 10, how many supercomputers use GPUs? Report the name of the supercomputers and their GPU vendor (Nvidia, AMD, ...) and model. 

8 supercomputers ues GPUs in top10.

|     name       | GPU vendor  |     model           | 
|----------------|:-----------:|:-------------------:|
| FRONTIER       | AMD         |	HPE Cray EX235a    |
| LUMI           | AMD         |	HPE Cray EX235a    |
|  Leonardo           | Nvidia        |	BullSequana XH2000    |
|  Summit          | Nvidia        |	IBM Power SystemAC922   |
|  Sierra           | Nvidia        |	IBM Power SystemS922LC   |
|  Perlmutter          | Nvidia        |	HPE Cray EX235a     |
|  Selene           | Nvidia        |	Nvidia |

#### 3. One main advantage of GPU is its power efficiency, which can be quantified by Performance/Power, e.g., throughput as in FLOPS per watt power consumption. Calculate the power efficiency for the top 10 supercomputers. (Hint: use the table in the first lecture)

## Exercise 2 - Query Nvidia GPU Compute Capability

#### 1. The screenshot of the output from running deviceQuery test in /1_Utilities.
![image](https://github.com/shiruimin123/DD2360GPU/blob/main/asssignment1/images/exercise2.jpg)

#### 2. What is the Compute Capability of your GPU device?
The compute capability of the GPU device, as shown in the output from deviceQuery, is "7.5."

#### 3. The screenshot of the output from running bandwidthTest test in /1_Utilities.
![image](https://github.com/shiruimin123/DD2360GPU/blob/main/asssignment1/images/bandwidthtest.jpg)

#### 4. How will you calculate the GPU memory bandwidth (in GB/s) using the output from deviceQuery? (Hint: memory bandwidth is typically determined by clock rate and bus width, and check what double date rate (DDR) may impact the bandwidth). Are they consistent with your results from bandwidthTest?
To calculate the GPU memory bandwidth (in GB/s) using the output from deviceQuery, the memory bandwidth can be calculated using the following formula:

Memory Bandwidth (GB/s) = Memory Clock Rate (GHz) x Memory Bus Width (bits) / 8

In the provided output, the memory clock rate is 5.001 MHz, and the memory bus width is 256 bits. The memory bandwidth should be:

Memory Bandwidth (GB/s) = 5.001 GHz x 256 bits / 8 = 160.032 GB/s

The calculated memory bandwidth is approximately 160.032 GB/s based on the output of deviceQuery.


## Exercise 3 - Rodinia CUDA benchmarks and Comparison with CPU

#### 1. Compile both OMP and CUDA versions of your selected benchmarks. Do you need to make any changes in Makefile?
The modifications we need to make to the makefile include paths and compute capabilities.

From exercise2 we found that the compute capability of Google Colab is "7.5". We specify the target architecture in the Makefile as "-arch sm_75" to support the code running on GPUs with sm_75 computing capabilities.

#### 2. Ensure the same input problem is used for OMP and CUDA versions. Report and compare their execution time. 
particlefilter[CUDA]

![image](https://github.com/shiruimin123/DD2360GPU/blob/main/asssignment1/images/particlefilter_cuda.jpg)

particlefilter[OpenMP]

![image](https://github.com/shiruimin123/DD2360GPU/blob/main/asssignment1/images/particlefilter_openmp.jpg)

lavaMD(CUDA)

![image](https://github.com/shiruimin123/DD2360GPU/blob/main/asssignment1/images/labaMD_cuda.jpg)

lavaMD(OpenMP)

![image](https://github.com/shiruimin123/DD2360GPU/blob/main/asssignment1/images/lavaMD_openmp.jpg)
#### 3. Do you observe expected speedup on GPU compared to CPU? Why or Why not?
In the benchmark we used, whether particlefilter or lavaMD, we can observe obvious acceleration.

## Exercise 4 - Run a HelloWorld on AMD GPU

#### 1. How do you launch the code on GPU on Dardel supercomputer?

#### 2. Include a screenshot of your output from Dardel
