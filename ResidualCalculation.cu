#include <cub/cub.cuh>
#include <cuda.h>

#include "Utilities.cuh"

#include <iostream>

#define BLOCKSIZE			256
#define ITEMS_PER_THREAD	8

const int N = 4096;

/******************************/
/* TRANSFORM REDUCTION KERNEL */
/******************************/
__global__ void TransformSumKernel(const float * __restrict__ indata1, const float * __restrict__ indata2, float * __restrict__ outdata) {
	
	unsigned int tid = threadIdx.x + blockIdx.x * gridDim.x;
	
	// --- Specialize BlockReduce for type float. 
	typedef cub::BlockReduce<float, BLOCKSIZE * ITEMS_PER_THREAD> BlockReduceT;

	__shared__ typename BlockReduceT::TempStorage  temp_storage;

	float result;
	if(tid < N) result = BlockReduceT(temp_storage).Sum((indata1[tid] - indata2[tid]) * (indata1[tid] - indata2[tid]));

	if(threadIdx.x == 0) atomicAdd(outdata, result);
  
	return;
}

/********/
/* MAIN */
/********/
int main() {
	
	// --- Allocate host side space for 
	float *h_data1		= (float *)malloc(N * sizeof(float));
	float *h_data2		= (float *)malloc(N * sizeof(float));
	float *h_result		= (float *)malloc(sizeof(float));

	float *d_data1;		gpuErrchk(cudaMalloc(&d_data1, N * sizeof(float)));
	float *d_data2;		gpuErrchk(cudaMalloc(&d_data2, N * sizeof(float)));
	float *d_result;	gpuErrchk(cudaMalloc(&d_result, sizeof(float)));
	
    for (int i = 0; i < N; i++) {
		h_data1[i] = 1.f;
		h_data2[i] = 3.f;
	}

    gpuErrchk(cudaMemcpy(d_data1, h_data1, N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_data2, h_data2, N * sizeof(float), cudaMemcpyHostToDevice));
    
	TransformSumKernel<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_data1, d_data2, d_result);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    	
	gpuErrchk(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
	std::cout << "output: ";
    std::cout << h_result[0];
    std::cout << std::endl;

	gpuErrchk(cudaFree(d_data1));
	gpuErrchk(cudaFree(d_data2));
	gpuErrchk(cudaFree(d_result));
    
	return 0;
}
