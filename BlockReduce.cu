#include <cub/cub.cuh>
#include <cuda.h>

#include "Utilities.cuh"

#include <iostream>

#define BLOCKSIZE	32

const int N = 1024;

/**************************/
/* BLOCK REDUCTION KERNEL */
/**************************/
__global__ void sum(const float * __restrict__ indata, float * __restrict__ outdata) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// --- Specialize BlockReduce for type float. 
	typedef cub::BlockReduce<float, BLOCKSIZE> BlockReduceT; 

	// --- Allocate temporary storage in shared memory 
	__shared__ typename BlockReduceT::TempStorage temp_storage; 

	float result;
	if(tid < N) result = BlockReduceT(temp_storage).Sum(indata[tid]);

	// --- Update block reduction value
	if(threadIdx.x == 0) outdata[blockIdx.x] = result;
	
	return;  
}

/********/
/* MAIN */
/********/
int main() {
	
	// --- Allocate host side space for 
	float *h_data		= (float *)malloc(N * sizeof(float));
	float *h_result		= (float *)malloc((N / BLOCKSIZE) * sizeof(float));

	float *d_data;		gpuErrchk(cudaMalloc(&d_data, N * sizeof(float)));
	float *d_result;	gpuErrchk(cudaMalloc(&d_result, (N / BLOCKSIZE) * sizeof(float)));
	
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    gpuErrchk(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
	sum<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_data, d_result);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    	
	gpuErrchk(cudaMemcpy(h_result, d_result, (N / BLOCKSIZE) * sizeof(float), cudaMemcpyDeviceToHost));
    
	std::cout << "output: ";
    for(int i = 0; i < (N / BLOCKSIZE); i++) std::cout << h_result[i] << " ";
    std::cout << std::endl;

	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_result));
    
	return 0;
}
