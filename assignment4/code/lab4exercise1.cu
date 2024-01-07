#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double
//#define nStreams 4
// GPU kernel for vector addition
//@@ Insert code to implement vector addition here
__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len,int offset) {
  int i = offset + blockIdx.x * blockDim.x + threadIdx.x;

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
  int nStreams;
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
    nStreams = atoi(argv[1]);    
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
  cudaHostAlloc(&hostInput1, inputLength * sizeof(DataType),cudaHostAllocDefault);
  cudaHostAlloc(&hostInput2, inputLength * sizeof(DataType),cudaHostAllocDefault);
  cudaHostAlloc(&hostOutput, inputLength * sizeof(DataType),cudaHostAllocDefault);
  resultRef  = (DataType*)malloc(inputLength * sizeof(DataType));
  double cpu_start = cpuSecond();
//@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = rand()/(DataType)RAND_MAX;
    hostInput2[i] = rand()/(DataType)RAND_MAX;
    resultRef[i]  = hostInput1[i] + hostInput2[i];
  }
  double cpu_end = cpuSecond();
  printf("CPU Execution Time: %f seconds\n", cpu_end - cpu_start);
//@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
//@@ Initialize the 1D grid and block dimensions here
  int Db = 1024;
  int Dg = (S_seg + Db - 1) / Db;

//@@ Insert code to below to Copy memory to the GPU here
double start = cpuSecond();
  for (int i = 0; i < nStreams; i++)  
  {
    int offset = i*S_seg;
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice,stream[i]); 
    vecAdd<<<Dg, Db,0,stream[i]>>>(deviceInput1, deviceInput2, deviceOutput, inputLength , offset);
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
  for (int i = 0; i < nStreams; ++i)
    cudaStreamDestroy( stream[i] );
//@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

//@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  free(resultRef);

  return 0;
}
