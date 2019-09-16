/*
 * CUDA convolutional neural net
 */

#include <iostream>
#include "ee155_utils.hxx"
#include "matrix.hxx"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
using namespace std;

const int BS=32;		// The blocks are BS x BS.

///////////////////////////////
// This is the CUDA kernel function for you to write.
//////////////////////////////
__global__ void CNN(float *d_inp, float *d_f, float *d_out, int FilterSize, int NBLK, int inputSize, size_t d_pitch, int NumOfRowsInFilter) {
	int rB = blockIdx.x;
	int cB = blockIdx.y;
	int rI = threadIdx.y;
	int cI = threadIdx.x;

	int OutBS = BS - FilterSize + 1;
	int r = rB * OutBS + rI;
	int c = cB * OutBS + cI;

	__shared__ float SI[BS][BS];
	if ((r < inputSize) && (c < inputSize))
		SI[rI][cI] = *(d_inp + r * inputSize + c);

	__syncthreads();

	if ((r < inputSize - FilterSize + 1) && (c < inputSize - FilterSize + 1)) {

		float filter = 0.f;
		float TempResult = 0.f;

		for (int fr = 0; fr < FilterSize; fr++) {
			for (int fc = 0; fc < FilterSize; fc++) {
				filter = *(d_f + fr * NumOfRowsInFilter + fc);
				if (rI + fr < BS && cI + fc < BS)
					TempResult += SI[rI + fr][cI + fc] * filter;
			}
		}
		if (rI < OutBS && cI < OutBS) {
			float* pElement = (float*)((char*)d_out + r * d_pitch) + c;
			*pElement = TempResult;
		}
	}
}


///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
//
void Matrix::CNN2(const Matrix &inp, const Matrix &f, int dummy) {
	auto start = start_time();

	// Allocate input matrix in device memory. It's a nice 2^N size, so don't
	// bother with cudaMallocPitch().
	assert(1 << inp._log2NColsAlc == inp._nCols);
	int numElem = inp.data.size(), sizeBytes = numElem * 4;
	float *d_inp = NULL;
	cudaError_t err = cudaMalloc((void **)&d_inp, sizeBytes);
	ERR_CHK(err, "Failed to allocate device matrix 'inp'");

	// Copy inp from host memory to device memory.
	err = cudaMemcpy(d_inp, inp.data.data(), sizeBytes, cudaMemcpyHostToDevice);
	ERR_CHK(err, "Failed to copy matrix inp from host to device");

	// Allocate device memory for filter. Again, don't bother with
	// cudaMallocPitch(); the filter is small, and Matrix has already picked 
	// a power of 2 columns
	float *d_f = NULL;
	sizeBytes = static_cast<int> (f.data.size()) * 4;
	err = cudaMalloc((void **)&d_f, sizeBytes);
	ERR_CHK(err, "Failed to allocate device matrix for the filter f");

	// Copy f from host memory to device memory.
	err = cudaMemcpy(d_f, f.data.data(), sizeBytes, cudaMemcpyHostToDevice);
	ERR_CHK(err, "Failed to copy matrix f from host to device");

	// Allocate device memory for the output matrix.
	float *d_out = NULL;
	size_t spitch;
	err = cudaMallocPitch((void **)&d_out, &spitch, 4 * this->N(), this->N());
	ERR_CHK(err, "Failed to allocate device matrix 'out'");
	long int time1 = delta_usec(start);

	// Launch the CUDA Kernel
	start = start_time();
	int FilterSize = f.nCols();
	int NBLK = ceil(this->N() / (float) (BS - FilterSize + 1));
	int NumOfRowsInFilter = (pow(2, (int)log2(FilterSize)) < FilterSize) ? (pow(2, (int)log2(FilterSize) + 1)) : FilterSize;
	dim3 thBlocks(NBLK, NBLK), threads(BS, BS);
	CNN <<<thBlocks,threads>>> (d_inp, d_f, d_out, FilterSize, NBLK, inp.nRows(), spitch, NumOfRowsInFilter);
	err = cudaGetLastError();
	ERR_CHK(err, "Failed to launch or finish CNN_kernel");
	long int time2 = delta_usec(start);

	// Copy the result from device memory to host memory.
	start = start_time();
	err = cudaMemcpy2D(this->data.data(), spitch, d_out, spitch, 4 * this->N(), this->N(), cudaMemcpyDeviceToHost);
	ERR_CHK(err, "Failed to copy result from device to host");

	err = cudaFree(d_inp);
	ERR_CHK(err, "Failed to free CUDA matrix inp");
	err = cudaFree(d_f);
	ERR_CHK(err, "Failed to free CUDA matrix f");
	err = cudaFree(d_out);
	ERR_CHK(err, "Failed to free CUDA matrix out");

	long int time3 = delta_usec(start);
	LOG("\tCUDA CNN took " << (time1 + time2 + time3) / 1000000.0 << "sec; " << (time1 / 1000000.0) << "s copy to, "
		<< (time2 / 1000000.0) << "s for computation, " << (time3 / 1000000.0) << "s copy back");
}
