#include <cub/cub.cuh>
#include <cuda.h>

#include "Utilities.cuh"

#include <iostream>

#define WARPSIZE	32
#define BLOCKSIZE	256

const int N = 1024;

/*************************/
/* WARP REDUCTION KERNEL */
/*************************/
__global__ void sum(const float * __restrict__ indata, float * __restrict__ outdata) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int warp_id = threadIdx.x / WARPSIZE;

	// --- Specialize WarpReduce for type float. 
	typedef cub::WarpReduce<float, WARPSIZE> WarpReduce;
    
    // --- Allocate WarpReduce shared memory for (N / WARPSIZE) warps
 	__shared__ typename WarpReduce::TempStorage temp_storage[BLOCKSIZE / WARPSIZE];
    
	float result;
	if(tid < N) result = WarpReduce(temp_storage[warp_id]).Sum(indata[tid]);

	if(tid % WARPSIZE == 0) outdata[tid / WARPSIZE] = result;
}

/********/
/* MAIN */
/********/
int main() {
	
	// --- Allocate host side space for 
	float *h_data		= (float *)malloc(N * sizeof(float));
	float *h_result		= (float *)malloc((N / WARPSIZE) * sizeof(float));

	float *d_data;		gpuErrchk(cudaMalloc(&d_data, N * sizeof(float)));
	float *d_result;	gpuErrchk(cudaMalloc(&d_result, (N / WARPSIZE) * sizeof(float)));
	
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    gpuErrchk(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
	sum<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_data, d_result);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    	
	gpuErrchk(cudaMemcpy(h_result, d_result, (N / WARPSIZE) * sizeof(float), cudaMemcpyDeviceToHost));
    
	std::cout << "output: ";
    for(int i = 0; i < (N / WARPSIZE); i++) std::cout << h_result[i] << " ";
    std::cout << std::endl;

	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_result));
    
	return 0;
}
