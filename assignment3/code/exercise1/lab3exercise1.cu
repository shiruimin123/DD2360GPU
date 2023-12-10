#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define BIN_COUNT 127

#define DataType unsigned int

__global__ void histogram_kernel(DataType *input, DataType *bins,
                  DataType num_elements,
                  DataType num_bins) {
  __shared__ unsigned int Bins[NUM_BINS];
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  /*
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    Bins[i] = 0;
  */
  for(int i = 0; i < NUM_BINS/1024; i++){
        Bins[i * 1024 + threadIdx.x] = 0;
    }
  
  __syncthreads();
// Calculate global thread ID
   if (idx < num_elements){
    atomicAdd(&Bins[input[idx]],1);
    }
   __syncthreads();
  for(int i = 0; i < NUM_BINS/1024; i++){
    if(Bins[i * 1024 + threadIdx.x] != 0){
      atomicAdd(&(bins[i * 1024 + threadIdx.x]),Bins[i * 1024 + threadIdx.x]);
    }
  }
  /*
   if(threadIdx.x == 0){
    for(int i = 0; i < NUM_BINS; i++){
      atomicAdd(&(bins[i]),Bins[i]);
    }
  }
  */
}
  double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
  }

  
//------------------------------------------------------------------------------
//@@ Insert code below to clean up bins that saturate at 127
__global__ void convert_kernel(DataType *bins, DataType num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_bins) {
        if (bins[tid] > BIN_COUNT) {
            bins[tid] = BIN_COUNT;
        }
    }
}

int main(int argc, char **argv) {
    int inputLength;
    DataType *hostInput;
    DataType *hostBins;
    DataType *resultRef;
    DataType *deviceInput;
    DataType *deviceBins;

//@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);

//@@ Insert code below to allocate Host memory for input and output
    hostInput = (DataType *)malloc(inputLength * sizeof(DataType));
    hostBins = (DataType *)malloc(NUM_BINS * sizeof(DataType));
    resultRef = (DataType *)malloc(NUM_BINS * sizeof(DataType));

//@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, NUM_BINS - 1);
    for (int i = 0; i < inputLength; ++i) {
        hostInput[i] = distribution(generator);
    }

//@@ Insert code below to create reference result in CPU
    for (int i = 0; i < NUM_BINS; ++i) {
        resultRef[i] = 0;
    }
    for (int i = 0; i < inputLength; ++i) {
        resultRef[hostInput[i]]++;
    }

//@@ Insert code below to allocate GPU memory here
    cudaMalloc((void **)&deviceInput, inputLength * sizeof(DataType));
    cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(DataType));

//@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

//@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(DataType));

//@@ Initialize the grid and block dimensions here
    dim3 blockDim1(1024);
    dim3 gridDim1((inputLength + blockDim1.x - 1) / blockDim1.x);

//@@ Launch the GPU Kernel here
    double gpu_histogram_start = cpuSecond();
    histogram_kernel<<<gridDim1, blockDim1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    cudaDeviceSynchronize();
    double gpu_histogram_end = cpuSecond();
    printf("Kernel Execution Time: %f seconds\n", gpu_histogram_end - gpu_histogram_start);

//@@ Initialize the second grid and block dimensions here
    dim3 blockDim2(1024);
    dim3 gridDim2((inputLength + blockDim2.x - 1) / blockDim2.x);

//@@ Launch the second GPU Kernel here
    convert_kernel<<<gridDim2, blockDim2>>>(deviceBins, NUM_BINS);
    cudaDeviceSynchronize();

//@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(DataType), cudaMemcpyDeviceToHost);

//@@ Insert code below to compare the output with the reference
    for (int i = 0; i < NUM_BINS; ++i) {
        if (hostBins[i] != resultRef[i]) {
            fprintf(stderr, "Mismatch at bin %d: Expected %u, Got %u\n", i, resultRef[i], hostBins[i]);
            //break;
        }
    }
//@@ Print histogram values
    FILE *fp = fopen("histogram.csv", "w");
    for (int i = 0; i < NUM_BINS; ++i)
    fprintf(fp, "%d\n", hostBins[i]);
    fclose(fp);
    
//@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

//@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}

