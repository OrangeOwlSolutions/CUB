#include <cub/cub.cuh>
#include <cuda.h>

#include "Utilities.cuh"

#include <iostream>

#define BLOCKSIZE			256

const int N = 4096;

/************************/
/* RASTRIGIN FUNCTIONAL */
/************************/
__device__ float rastrigin(float x) {
	
	return x * x - 10.0f * cosf(2.0f * x) + 10.0f;

}

/******************************/
/* TRANSFORM REDUCTION KERNEL */
/******************************/
__global__ void CostFunctionalCalculation(const float * __restrict__ indata, float * __restrict__ outdata) {
	
	unsigned int tid = threadIdx.x + blockIdx.x * gridDim.x;
	
	// --- Specialize BlockReduce for type float. 
	typedef cub::BlockReduce<float, BLOCKSIZE> BlockReduceT;

	__shared__ typename BlockReduceT::TempStorage  temp_storage;

	float result;
	if(tid < N) result = BlockReduceT(temp_storage).Sum(rastrigin(indata[tid]));

	if(threadIdx.x == 0) outdata[blockIdx.x] = result;
  
	return;
}

/********/
/* MAIN */
/********/
int main() {
	
	// --- Allocate host side space for 
	float *h_data		= (float *)malloc(N				  * sizeof(float));
	float *h_result		= (float *)malloc((N / BLOCKSIZE) * sizeof(float));

	float *d_data;		gpuErrchk(cudaMalloc(&d_data,   N               * sizeof(float)));
	float *d_result;	gpuErrchk(cudaMalloc(&d_result, (N / BLOCKSIZE) * sizeof(float)));
	
    for (int i = 0; i < N; i++) {
		h_data[i] = 1.f;
	}

    gpuErrchk(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
	CostFunctionalCalculation<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_data, d_result);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    	
	gpuErrchk(cudaMemcpy(h_result, d_result, (N / BLOCKSIZE) * sizeof(float), cudaMemcpyDeviceToHost));
    
	std::cout << "output: \n";
    for (int k = 0; k < N / BLOCKSIZE; k++)	std::cout << k << " " << h_result[k] << "\n";
    std::cout << std::endl;

	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_result));
    
	return 0;
}
