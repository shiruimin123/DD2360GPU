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

## Exercise 3 - Rodinia CUDA benchmarks and Comparison with CPU

#### 1. Compile both OMP and CUDA versions of your selected benchmarks. Do you need to make any changes in Makefile?
#### 2. Ensure the same input problem is used for OMP and CUDA versions. Report and compare their execution time. 

#### 3. Do you observe expected speedup on GPU compared to CPU? Why or Why not?

## Exercise 4 - Run a HelloWorld on AMD GPU

#### 1. How do you launch the code on GPU on Dardel supercomputer?

#### 2. Include a screenshot of your output from Dardel
