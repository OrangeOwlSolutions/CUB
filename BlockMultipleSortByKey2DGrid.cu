#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>

#include "Utilities.cuh"

using namespace cub;

/*******************************/
/* CUB BLOCKSORT KERNEL SHARED */
/*******************************/
template <int BLOCKSIZE_X, int BLOCKSIZE_Y, int ITEMS_PER_THREAD>
__global__ void shared_BlockSortKernel(float *d_valuesA, float *d_valuesB, int *d_keys, float *d_values_resultA, float *d_values_resultB, int *d_keys_result)
{
    // --- Shared memory allocation
	__shared__ float sharedMemoryArrayValuesA [BLOCKSIZE_X * BLOCKSIZE_Y * ITEMS_PER_THREAD];
	__shared__ float sharedMemoryArrayValuesB [BLOCKSIZE_X * BLOCKSIZE_Y * ITEMS_PER_THREAD];
	__shared__ int   sharedMemoryArrayKeys    [BLOCKSIZE_X * BLOCKSIZE_Y * ITEMS_PER_THREAD];
	__shared__ int   sharedMemoryHelperIndices[BLOCKSIZE_X * BLOCKSIZE_Y * ITEMS_PER_THREAD];

	// --- Specialize BlockStore and BlockRadixSort collective types
	typedef cub::BlockRadixSort	<int , BLOCKSIZE_X, ITEMS_PER_THREAD, int, 4, false, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, BLOCKSIZE_Y>	BlockRadixSortT;
    
	// --- Allocate type-safe, repurposable shared memory for collectives
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

	int block_offset = blockIdx.x * (BLOCKSIZE_X * BLOCKSIZE_Y * ITEMS_PER_THREAD);

	// --- Load data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
		sharedMemoryArrayValuesA [(threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k] = d_valuesA[block_offset + (threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k];
		sharedMemoryArrayValuesB [(threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k] = d_valuesB[block_offset + (threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k];
		sharedMemoryArrayKeys    [(threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k] = d_keys   [block_offset + (threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k];
		sharedMemoryHelperIndices[(threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k] =                          (threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k ;
	}
	__syncthreads();

    // --- Collectively sort the keys
    BlockRadixSortT(temp_storage).SortBlockedToStriped(*static_cast<int(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryArrayKeys     + ((threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD))),
		                                               *static_cast<int(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryHelperIndices + ((threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD))));
    __syncthreads();

	// --- Write data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
		d_values_resultA[block_offset + (threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k] = sharedMemoryArrayValuesA[sharedMemoryHelperIndices[(threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k]];
		d_values_resultB[block_offset + (threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k] = sharedMemoryArrayValuesB[sharedMemoryHelperIndices[(threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k]];
		d_keys_result   [block_offset + (threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k] = sharedMemoryArrayKeys                             [(threadIdx.y * BLOCKSIZE_X + threadIdx.x) * ITEMS_PER_THREAD + k];
	}
}

/********/
/* MAIN */
/********/
int main() {

	const int blockSize_x		= 2;
	const int blockSize_y		= 4;
	const int numElemsPerArray  = blockSize_x * blockSize_y;
	const int numArrays			= 4;
	const int N					= numArrays * numElemsPerArray;
	const int numElemsPerThread = numElemsPerArray / (blockSize_x * blockSize_y);

	const int RANGE				= N * numElemsPerThread;

	// --- Allocating and initializing the data on the host
	float *h_valuesA	= (float *)malloc(N * sizeof(float));
	float *h_valuesB	= (float *)malloc(N * sizeof(float));
	int *h_keys			= (int *)  malloc(N * sizeof(int));
    for (int i = 0 ; i < N; i++) {
		h_valuesA[i] = rand() % RANGE;
		h_valuesB[i] = rand() % RANGE;
		h_keys[i]	 = rand() % RANGE;
	}

	printf("Original\n\n");
	for (int k = 0; k < numArrays; k++) 
		for (int i = 0; i < numElemsPerArray; i++)
			printf("Array nr. %i; Element nr. %i; Key %i; Value A %f; Value B %f\n", k, i, h_keys[k * numElemsPerArray + i], h_valuesA[k * numElemsPerArray + i], h_valuesB[k * numElemsPerArray + i]);

	// --- Allocating the results on the host
	float *h_values_resultA  = (float *)malloc(N * sizeof(float));
	float *h_values_resultB  = (float *)malloc(N * sizeof(float));
	float *h_values_result2  = (float *)malloc(N * sizeof(float));
	int   *h_keys_result1	 = (int *)  malloc(N * sizeof(int));
	int   *h_keys_result2	 = (int *)  malloc(N * sizeof(int));
        
    // --- Allocating space for data and results on device
    float *d_valuesA;			gpuErrchk(cudaMalloc((void **)&d_valuesA,		 N * sizeof(float)));
    float *d_valuesB;			gpuErrchk(cudaMalloc((void **)&d_valuesB,		 N * sizeof(float)));
    int   *d_keys;				gpuErrchk(cudaMalloc((void **)&d_keys,			 N * sizeof(int)));
    float *d_values_resultA;	gpuErrchk(cudaMalloc((void **)&d_values_resultA, N * sizeof(float)));
    float *d_values_resultB;	gpuErrchk(cudaMalloc((void **)&d_values_resultB, N * sizeof(float)));
    float *d_values_result2;	gpuErrchk(cudaMalloc((void **)&d_values_result2, N * sizeof(float)));
    int   *d_keys_result1;		gpuErrchk(cudaMalloc((void **)&d_keys_result1,   N * sizeof(int)));
    int   *d_keys_result2;		gpuErrchk(cudaMalloc((void **)&d_keys_result2,   N * sizeof(int)));

	// --- BlockSortKernel with shared
	gpuErrchk(cudaMemcpy(d_valuesA, h_valuesA, N * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_valuesB, h_valuesB, N * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_keys,   h_keys,   N * sizeof(int),   cudaMemcpyHostToDevice));
    shared_BlockSortKernel<blockSize_x, blockSize_y, numElemsPerThread><<<numArrays, numElemsPerArray / numElemsPerThread>>>(d_valuesA, d_valuesB, d_keys, d_values_resultA, d_values_resultB, d_keys_result1); 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());    
	gpuErrchk(cudaMemcpy(h_values_resultA, d_values_resultA, N * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_values_resultB, d_values_resultB, N * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_keys_result1,   d_keys_result1,   N * sizeof(int),   cudaMemcpyDeviceToHost));
    
	printf("\n\nBlockSortKernel using shared memory\n\n");
	for (int k = 0; k < numArrays; k++) 
		for (int i = 0; i < numElemsPerArray; i++)
			printf("Array nr. %i; Element nr. %i; Key %i; Value %f; Value %f\n", k, i, h_keys_result1[k * numElemsPerArray + i], h_values_resultA[k * numElemsPerArray + i], h_values_resultB[k * numElemsPerArray + i]);

	return 0;
}
