# Assignment I:

## Exercise 1 - Reflection on GPU-accelerated Computing

#### 1. List the main differences between GPUs and CPUs in terms of architecture.
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

## Exercise 4 - Run a HelloWorld on AMD GPU

#### 1. How do you launch the code on GPU on Dardel supercomputer?

#### 2. Include a screenshot of your output from Dardel
