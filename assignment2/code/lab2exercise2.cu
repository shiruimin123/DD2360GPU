#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
//printf("hello");
if((row < numARows)&&(col < numBColumns)){
  DataType sum = 0.0;
  for(int i = 0; i < numAColumns; i++){
    sum += A[row * numAColumns + i] * B[i * numBColumns + col];
  }
  C[row * numBColumns + col] = sum;
//printf("Result in Kernel (%f)", C[row * numBColumns + col]);
 }
}
double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
int main(int argc, char **argv) {

    DataType *hostA;       // The A matrix
    DataType *hostB;       // The B matrix
    DataType *hostC;       // The output C matrix
    DataType *resultRef;     // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;        // number of rows in the matrix A
    int numAColumns;      // number of columns in the matrix A
    int numBRows;       // number of rows in the matrix B
    int numBColumns;     // number of columns in the matrix B
    int numCRows;
    int numCColumns;

//@@ Insert code below to read in numARows, numAColumns, numBRows, numBColumns from args
//    numARows = 3;
//    numAColumns = 4;
//    numBRows = 4;
//    numBColumns = 5;
//    numCRows = numARows;
//    numCColumns = numBColumns;

//    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
if(argc != 5) {
  printf("ERROR: Exactly four input parameters are required to run the program (%d != 4).\n", argc);
  exit(1);
}
      numARows = atoi(argv[1]);
      numAColumns = atoi(argv[2]);
      numBRows = atoi(argv[3]);
      numBColumns = atoi(argv[4]);
      numCRows = numARows;
      numCColumns = numBColumns;
 printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
 if(numAColumns != numBRows){
  printf("ERROR: Matrix A must have the same number of columns as the number of rows of matrix B (%d != %d).\n", numAColumns, numBRows);
  return 0;
}

//@@ Insert code below to allocate Host memory for input and output
hostA   = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
hostB   = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
hostC   = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

//@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
for(int i = 0; i < numARows; i++){
  for(int j = 0; j < numAColumns; j++){
    hostA[i * numAColumns + j] = rand()/(DataType)RAND_MAX;
  }
}
for (int i = 0; i < numARows; i++){
  for (int j = 0; j < numAColumns; j++){
    hostB[i * numBColumns + j] = rand()/(DataType)RAND_MAX;
  }
}
// Compute the reference result on the CPU
for(int i = 0; i < numARows; i++){
  for(int j = 0; j < numBColumns; j++){
    DataType sum = 0.0;
    for(int k = 0; k < numAColumns; k++){
      sum += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
    }
  resultRef[i * numBColumns + j] = sum;
  }
}

//@@ Insert code below to allocate GPU memory here
cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));

//@@ Insert code to below to Copy memory to the GPU here
cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);

//@@ Initialize the grid and block dimensions here
//int Dbx = 32;    //Can be adjust according to the architecture of GPU
//int Dby = 32;
//int Dgx = (numBColumns + Dbx - 1)/Dbx;
//int Dgy = (numARows + Dby - 1)/Dby;
dim3 dimBlock(16,16);
dim3 dimGrid((numBColumns + dimBlock.x - 1) / dimBlock.x, (numARows + dimBlock.y - 1) / dimBlock.y);

//@@ Launch the GPU Kernel here
double gpu_start = cpuSecond();
gemm<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numAColumns, numBColumns);
//gemm<<<dim3(Dgx,Dgy),dim3(Dbx,Dby)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
cudaDeviceSynchronize();
double gpu_end = cpuSecond();
printf("Kernel Execution Time: %f seconds\n", gpu_end - gpu_start);

//@@ Copy the GPU memory back to the CPU here
cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);

//@@ Insert code below to compare the output with the reference
for(int i = 0; i < numCRows; i++){
  for(int j = 0; j < numCColumns; j++){
    if(abs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) > 1e-5){
      printf("Mismatch at element (%d, %d): %f != %f\n", i, j, hostC[i * numCColumns + j], resultRef[i * numCColumns + j]);
    }
  }
}
printf("Results match!\n");

//@@ Free the GPU memory here
cudaFree(deviceA);
cudaFree(deviceB);
cudaFree(deviceC);

//@@ Free the CPU memory here
free(hostA);
free(hostB);
free(hostC);
free(resultRef);

return 0;
}