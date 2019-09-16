#include <iostream>
#include "ee155_utils.hxx"
#include "matrix.hxx"

// For the CUDA runtime routines (prefixed with "cuda_")
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

const int BS = 32;	// The blocks are BS x BS.


///////////////////////////////
// This is the CUDA kernel function for you to write.
//
__global__ void mat_mult (float *d_A, float *d_B, float *d_C, int N) {
	int rB = blockIdx.x;
	int cB = blockIdx.y;
	int rI = threadIdx.y;
	int cI = threadIdx.x;
	
    __shared__ float SA[BS][BS], SB[BS][BS];
    //printf("In thread with r=(%d,%d) c=(%d,%d)\n", rB,rI,cB,cI);

	float TempResult = 0.f;
	for (int kB = 0; kB < N; kB++)
	{
		SA[rI][cI] = *(d_A + rB * BS * N * BS + rI * N * BS + kB * BS + cI);
		SB[rI][cI] = *(d_B + kB * BS * N * BS + rI * N * BS + cB * BS + cI);

		__syncthreads();
		for (int kI = 0; kI < BS; kI++)
		{
			TempResult += SA[rI][kI] * SB[kI][cI];
		}
		__syncthreads();
	}
	*(d_C + rB * BS * N * BS + rI * N * BS + cB * BS + cI) = TempResult;    // Store results back to device memory
}


///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
void Matrix::mpy1 (const Matrix &A, const Matrix &B, int BS) {

    // Copy A from host memory to device memory.
    int numElem=N()*N(), sizeBytes = numElem*4;
    float *d_A = NULL;
    cudaError_t err = cudaMalloc((void **)&d_A, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix A");

    err = cudaMemcpy (d_A, A.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix A from host to device");

    // Allocate device memory for B.
	float* d_B = NULL;                           // pointer into GPU memory
	err = cudaMalloc((void**)&d_B, sizeBytes);
	ERR_CHK(err, "Failed to allocate device matrix B");

    // Copy B from host memory to device memory.
	err = cudaMemcpy(d_B, B.data.data(), sizeBytes, cudaMemcpyHostToDevice);
	ERR_CHK(err, "Failed to copy matrix B from host to device");

    // Allocate device memory for C.
	float* d_C = NULL;                           // pointer into GPU memory
	err = cudaMalloc((void**)&d_C, sizeBytes);
	ERR_CHK(err, "Failed to allocate device matrix C");

    // Launch the CUDA Kernel
	int NBLK = this->N() / BS;
	dim3 thBlocks(NBLK, NBLK), threads(BS, BS);

	mat_mult <<<thBlocks, threads>>> (d_A, d_B, d_C, NBLK);

    // Copy the result from device memory to host memory.
	err = cudaMemcpy(this->data.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost);
	ERR_CHK(err, "Failed to copy data back from GPU to CPU");

    // Free device memory.
    err = cudaFree(d_A);
    ERR_CHK (err, "Failed to free CUDA matrix A");
	err = cudaFree(d_B);
	ERR_CHK(err, "Failed to free CUDA matrix B");
	err = cudaFree(d_C);
	ERR_CHK(err, "Failed to free CUDA matrix C");
}
