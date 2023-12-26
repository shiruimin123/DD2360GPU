#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double
#define nStreams 4
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

  const int S_seg = inputLength / nStreams;
  const int streamBytes = S_seg * sizeof(DataType);


// create stream
  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]); 


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
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
//@@ Initialize the 1D grid and block dimensions here
  int Db = 1024;
  int Dg = (inputLength + Db - 1) / Db;

//@@ Insert code to below to Copy memory to the GPU here
double start = cpuSecond();
  for (int i = 0; i < nStreams; ++i)  {
    int offset = i*streamSize;
  //double h2d_start = cpuSecond();
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice,stream[i]);
  }
  //double h2d_end = cpuSecond();
  //printf("Data copy from host to device: %f seconds\n", h2d_end - h2d_start);



//@@ Launch the GPU Kernel here
 // double gpu_start = cpuSecond();
  for (int i = 0; i < nStreams; ++i)  
    vecAdd<<<Dg, Db,0,stream[i]>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
 // cudaDeviceSynchronize();
  //double gpu_end = cpuSecond();
 // printf("Kernel Execution Time: %f seconds\n", gpu_end - gpu_start);

//@@ Copy the GPU memory back to the CPU here
//  double d2h_start = cpuSecond();
for (int i = 0; i < nStreams; ++i)  {
  int offset = i*streamSize;
  cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost,stream[i]);
}

cudaDeviceSynchronize();
double end = cpuSecond();
printf("GPU Execution Time: %f seconds\n", end - start);
//  double d2h_end = cpuSecond();
//  printf("Data copy from device to host: %f seconds\n", d2h_end - d2h_start);
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