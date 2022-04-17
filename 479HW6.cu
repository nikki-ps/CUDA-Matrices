#include <stdio.h>
#include <cuda.h>
//Kernel to initialize matrix and compute number of even values in the matrix
__global__ void dkernel(unsigned *matrix, int *evenNum) {
    //initializing the matrix 
    //matrix spots are filled with their respective id's 
    unsigned id = threadIdx.x * blockDim.y + threadIdx.y;
    matrix[id] = id;
    //if the matrix value has no remainder when divided by 2 it is even
    //atomicAdd guarentees add operation is performed without interference from other threads. 
    //No other thread can access this address until the operation is complete
    if((matrix[id] % 2) == 0) {
        atomicAdd(evenNum, 1);
    }
}
//kernel to compute the square of a given matrix
__global__ void square ( unsigned *matrix, unsigned *result, unsigned matrixsize) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned jj = 0; jj < matrixsize; ++jj) 
    {
        for (unsigned kk = 0; kk < matrixsize; ++kk) 
        {
            result[id * matrixsize + jj] += matrix[id * matrixsize + kk] *
            matrix[kk * matrixsize + jj];
        }
    }
}
#define N 8
#define M 8
int main() {
    //*****EXERCISE 1*****
    dim3 block(N, M, 1);
    unsigned *matrix, *hmatrix, *resultGPU, *sqauredResult;
    int *evenNum;
    int count;
    //memory is allocated
    cudaMalloc(&matrix, N * M * sizeof(unsigned));
    cudaMalloc(&evenNum, sizeof(int));
    hmatrix = (unsigned *)malloc(N * M * sizeof(unsigned));
    //kernel called
    dkernel<<<1, block>>>(matrix, evenNum);
    //transfer between host memory and device memory
    cudaMemcpy(hmatrix, matrix, N * M * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, evenNum, sizeof(int), cudaMemcpyDeviceToHost);
    //matrix printed
    printf("Original Matrix:\n");
    for (unsigned ii = 0; ii < N; ++ii) 
    {
        for (unsigned jj = 0; jj < M; ++jj) 
        {
            printf("%2d ", hmatrix[ii * M + jj]);
        }
        printf("\n");
    }
    //number of even value in matrix printed
    printf("Number of even values in the NxN matrix: %d\n\n", count);
    

    //*****EXERCISE 2*****
    //memory is allocated
    sqauredResult = (unsigned *)malloc(N * M * sizeof(unsigned));
    cudaMalloc(&resultGPU, N * M * sizeof(unsigned));
    //kernel called
    square<<<1, N>>>(matrix, resultGPU, N);
    //trasnfer between host memory and device memory
    cudaMemcpy(sqauredResult, resultGPU, N * M * sizeof(unsigned), cudaMemcpyDeviceToHost);
    //squared matrix printed
    printf("Sqaure Matrix:\n");
    for (unsigned ii = 0; ii < N; ++ii) 
    {
        for (unsigned jj = 0; jj < N; ++jj) 
        { 
            printf("%2d ", sqauredResult[ii * N + jj]); 
        } 
        printf("\n"); 
    } 

    return 0;
}