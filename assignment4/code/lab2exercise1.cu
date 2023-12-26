#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

// GPU kernel for vector addition
//@@ Insert code to implement vector addition here
__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    out[i] = in1[i] + in2[i];
  }
}
//----------
//@@ Insert code to implement timer start
//@@ Insert code to implement timer stop
double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//----------
int main(int argc, char **argv) {
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

//@@ Insert code below to read in inputLength from args
  if (argc > 1) {
    inputLength = atoi(argv[1]);                           //ASCII to integer
  }
  printf("The input length is %d\n", inputLength);
  
//----------
// Allocate Host memory for input and output
  hostInput1 = (DataType*)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType*)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType*)malloc(inputLength * sizeof(DataType));
  resultRef  = (DataType*)malloc(inputLength * sizeof(DataType));

//@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = rand()/(DataType)RAND_MAX;
    hostInput2[i] = rand()/(DataType)RAND_MAX;
    resultRef[i]  = hostInput1[i] + hostInput2[i];
  }

//@@ Insert code below to allocate GPU memory here
  cudaMalloc((void**)&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc((void**)&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc((void**)&deviceOutput, inputLength * sizeof(DataType));

//@@ Insert code to below to Copy memory to the GPU here
  double h2d_start = cpuSecond();
  double start = cpuSecond();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  double h2d_end = cpuSecond();
  printf("Data copy from host to device: %f seconds\n", h2d_end - h2d_start);

//@@ Initialize the 1D grid and block dimensions here
  int Db = 1024;
  int Dg = (inputLength + Db - 1) / Db;

//@@ Launch the GPU Kernel here
  double gpu_start = cpuSecond();
  vecAdd<<<Dg, Db>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double gpu_end = cpuSecond();
  printf("Kernel Execution Time: %f seconds\n", gpu_end - gpu_start);

//@@ Copy the GPU memory back to the CPU here
  double d2h_start = cpuSecond();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  double d2h_end = cpuSecond();
  printf("Data copy from device to host: %f seconds\n", d2h_end - d2h_start);
  double end = cpuSecond();
  printf("GPU Execution: %f seconds\n", end - start);
//@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++) {
    if (fabs(hostOutput[i] - resultRef[i]) > 1e-5) {
      fprintf(stderr, "Mismatch at index %d: %f != %f\n", i, hostOutput[i], resultRef[i]);
      break;
    }
  }

//@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

//@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}