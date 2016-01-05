#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>

#include "Utilities.cuh"

using namespace cub;

/**********************************/
/* CUB BLOCKSORT KERNEL NO SHARED */
/**********************************/
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(float *d_values, int *d_keys, float *d_values_result, int *d_keys_result)
{
	// --- Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
	typedef cub::BlockLoad		<int*,   BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>	 BlockLoadIntT;
	typedef cub::BlockLoad		<float*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>	 BlockLoadFloatT;
	typedef cub::BlockStore		<int*,   BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE> BlockStoreIntT;
	typedef cub::BlockStore		<float*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE> BlockStoreFloatT;
	typedef cub::BlockRadixSort	<int ,   BLOCK_THREADS, ITEMS_PER_THREAD, float>				 BlockRadixSortT;

	// --- Allocate type-safe, repurposable shared memory for collectives
	__shared__ union {
		typename BlockLoadIntT		::TempStorage loadInt;
		typename BlockLoadFloatT	::TempStorage loadFloat;
		typename BlockStoreIntT		::TempStorage storeInt;
		typename BlockStoreFloatT	::TempStorage storeFloat;
		typename BlockRadixSortT	::TempStorage sort;
    } temp_storage;

	// --- Obtain this block's segment of consecutive keys (blocked across threads)
	int   thread_keys[ITEMS_PER_THREAD];
	float thread_values[ITEMS_PER_THREAD];
	int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

	BlockLoadIntT(temp_storage.loadInt).Load(d_keys   + block_offset, thread_keys);
	BlockLoadFloatT(temp_storage.loadFloat).Load(d_values + block_offset, thread_values);
	__syncthreads(); 

	// --- Collectively sort the keys
	BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(thread_keys, thread_values);
	__syncthreads(); 

	// --- Store the sorted segment
	BlockStoreIntT(temp_storage.storeInt).Store(d_keys_result   + block_offset, thread_keys);
	BlockStoreFloatT(temp_storage.storeFloat).Store(d_values_result + block_offset, thread_values);
  
}

/*******************************/
/* CUB BLOCKSORT KERNEL SHARED */
/*******************************/
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void shared_BlockSortKernel(float *d_values, int *d_keys, float *d_values_result, int *d_keys_result)
{
    // --- Shared memory allocation
	__shared__ float sharedMemoryArrayValues[BLOCK_THREADS * ITEMS_PER_THREAD];
	__shared__ int   sharedMemoryArrayKeys[BLOCK_THREADS * ITEMS_PER_THREAD];

	// --- Specialize BlockStore and BlockRadixSort collective types
	typedef cub::BlockRadixSort	<int , BLOCK_THREADS, ITEMS_PER_THREAD, float>	BlockRadixSortT;
    
	// --- Allocate type-safe, repurposable shared memory for collectives
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

	int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

	// --- Load data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
		sharedMemoryArrayValues[threadIdx.x * ITEMS_PER_THREAD + k] = d_values[block_offset + threadIdx.x * ITEMS_PER_THREAD + k];
		sharedMemoryArrayKeys[threadIdx.x * ITEMS_PER_THREAD + k]   = d_keys[block_offset + threadIdx.x * ITEMS_PER_THREAD + k];
	}
	__syncthreads();

    // --- Collectively sort the keys
    BlockRadixSortT(temp_storage).SortBlockedToStriped(*static_cast<int(*)  [ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryArrayKeys   + (threadIdx.x * ITEMS_PER_THREAD))),
		                                               *static_cast<float(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryArrayValues + (threadIdx.x * ITEMS_PER_THREAD))));
    __syncthreads();

	// --- Write data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
		d_values_result[block_offset + threadIdx.x * ITEMS_PER_THREAD + k] = sharedMemoryArrayValues[threadIdx.x * ITEMS_PER_THREAD + k];
		d_keys_result  [block_offset + threadIdx.x * ITEMS_PER_THREAD + k] = sharedMemoryArrayKeys  [threadIdx.x * ITEMS_PER_THREAD + k];
	}
}

/********/
/* MAIN */
/********/
int main() {

	const int numElemsPerArray  = 8;
	const int numArrays			= 4;
	const int N					= numArrays * numElemsPerArray;
	const int numElemsPerThread = 4;

	const int RANGE				= N * numElemsPerThread;

	// --- Allocating and initializing the data on the host
	float *h_values	= (float *)malloc(N * sizeof(float));
	int *h_keys		= (int *)  malloc(N * sizeof(int));
    for (int i = 0 ; i < N; i++) {
		h_values[i] = rand() % RANGE;
		h_keys[i]	= rand() % RANGE;
	}

	printf("Original\n\n");
	for (int k = 0; k < numArrays; k++) 
		for (int i = 0; i < numElemsPerArray; i++)
			printf("Array nr. %i; Element nr. %i; Key %i; Value %f\n", k, i, h_keys[k * numElemsPerArray + i], h_values[k * numElemsPerArray + i]);

	// --- Allocating the results on the host
	float *h_values_result1 = (float *)malloc(N * sizeof(float));
	float *h_values_result2 = (float *)malloc(N * sizeof(float));
	int   *h_keys_result1	= (int *)  malloc(N * sizeof(int));
	int   *h_keys_result2	= (int *)  malloc(N * sizeof(int));
        
    // --- Allocating space for data and results on device
    float *d_values;			gpuErrchk(cudaMalloc((void **)&d_values,		 N * sizeof(float)));
    int   *d_keys;				gpuErrchk(cudaMalloc((void **)&d_keys,			 N * sizeof(int)));
    float *d_values_result1;	gpuErrchk(cudaMalloc((void **)&d_values_result1, N * sizeof(float)));
    float *d_values_result2;	gpuErrchk(cudaMalloc((void **)&d_values_result2, N * sizeof(float)));
    int   *d_keys_result1;		gpuErrchk(cudaMalloc((void **)&d_keys_result1,   N * sizeof(int)));
    int   *d_keys_result2;		gpuErrchk(cudaMalloc((void **)&d_keys_result2,   N * sizeof(int)));

	// --- BlockSortKernel no shared
	gpuErrchk(cudaMemcpy(d_values, h_values, N * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_keys,   h_keys,   N * sizeof(int),   cudaMemcpyHostToDevice));
	BlockSortKernel<N / numArrays / numElemsPerThread, numElemsPerThread><<<numArrays, numElemsPerArray / numElemsPerThread>>>(d_values, d_keys, d_values_result1, d_keys_result1); 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());    
	gpuErrchk(cudaMemcpy(h_values_result1, d_values_result1, N * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_keys_result1,   d_keys_result1,   N * sizeof(int),   cudaMemcpyDeviceToHost));
    
	printf("\n\nBlockSortKernel no shared\n\n");
	for (int k = 0; k < numArrays; k++) 
		for (int i = 0; i < numElemsPerArray; i++)
			printf("Array nr. %i; Element nr. %i; Key %i; Value %f\n", k, i, h_keys_result1[k * numElemsPerArray + i], h_values_result1[k * numElemsPerArray + i]);
	
	// --- BlockSortKernel with shared
	gpuErrchk(cudaMemcpy(d_values, h_values, N * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_keys,   h_keys,   N * sizeof(int),   cudaMemcpyHostToDevice));
    shared_BlockSortKernel<N / numArrays / numElemsPerThread, numElemsPerThread><<<numArrays, numElemsPerArray / numElemsPerThread>>>(d_values, d_keys, d_values_result2, d_keys_result2); 
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());    
	gpuErrchk(cudaMemcpy(h_values_result2, d_values_result2, N * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_keys_result2,   d_keys_result2,   N * sizeof(int),   cudaMemcpyDeviceToHost));
    
	printf("\n\nBlockSortKernel shared\n\n");
	for (int k = 0; k < numArrays; k++) 
		for (int i = 0; i < numElemsPerArray; i++)
			printf("Array nr. %i; Element nr. %i; Key %i; Value %f\n", k, i, h_keys_result2[k * numElemsPerArray + i], h_values_result2[k * numElemsPerArray + i]);

	return 0;
}
