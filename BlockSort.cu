#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>

#include "Utilities.cuh"

using namespace cub;

/**********************************/
/* CUB BLOCKSORT KERNEL NO SHARED */
/**********************************/
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
	// --- Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
	typedef cub::BlockLoad		<int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>	BlockLoadT;
	typedef cub::BlockStore		<int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE>	BlockStoreT;
	typedef cub::BlockRadixSort	<int , BLOCK_THREADS, ITEMS_PER_THREAD>							BlockRadixSortT;

	// --- Allocate type-safe, repurposable shared memory for collectives
	__shared__ union {
		typename BlockLoadT		::TempStorage load;
		typename BlockStoreT	::TempStorage store;
		typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

	// --- Obtain this block's segment of consecutive keys (blocked across threads)
	int thread_keys[ITEMS_PER_THREAD];
	int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

	BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);
	__syncthreads(); 

	// --- Collectively sort the keys
	BlockRadixSortT(temp_storage.sort).Sort(thread_keys);
	__syncthreads(); 

	// --- Store the sorted segment
	BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
  
}

/*******************************/
/* CUB BLOCKSORT KERNEL SHARED */
/*******************************/
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void shared_BlockSortKernel(int *d_in, int *d_out)
{
    // --- Shared memory allocation
	__shared__ int sharedMemoryArray[BLOCK_THREADS * ITEMS_PER_THREAD];

	// --- Specialize BlockStore and BlockRadixSort collective types
	typedef cub::BlockRadixSort	<int , BLOCK_THREADS, ITEMS_PER_THREAD>	BlockRadixSortT;
    
	// --- Allocate type-safe, repurposable shared memory for collectives
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

	int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

	// --- Load data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) sharedMemoryArray[threadIdx.x * ITEMS_PER_THREAD + k]  = d_in[block_offset + threadIdx.x * ITEMS_PER_THREAD + k];
	__syncthreads();

    // --- Collectively sort the keys
    BlockRadixSortT(temp_storage).Sort(*static_cast<int(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryArray + (threadIdx.x * ITEMS_PER_THREAD))));
    __syncthreads();

	// --- Write data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) d_out[block_offset + threadIdx.x * ITEMS_PER_THREAD + k] = sharedMemoryArray[threadIdx.x * ITEMS_PER_THREAD + k];
	
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
	int *h_data	= (int *)malloc(N * sizeof(int));
    for (int i = 0 ; i < N; i++) h_data[i] = rand() % RANGE;

	// --- Allocating the results on the host
	int *h_result1 = (int *)malloc(N * sizeof(int));
	int *h_result2 = (int *)malloc(N * sizeof(int));
        
    // --- Allocating space for data and results on device
    int *d_in;		gpuErrchk(cudaMalloc((void **)&d_in,   N * sizeof(int)));
    int *d_out1;	gpuErrchk(cudaMalloc((void **)&d_out1, N * sizeof(int)));
    int *d_out2;	gpuErrchk(cudaMalloc((void **)&d_out2, N * sizeof(int)));

	// --- BlockSortKernel no shared
	gpuErrchk(cudaMemcpy(d_in, h_data, N*sizeof(int), cudaMemcpyHostToDevice));
	BlockSortKernel<N / numArrays / numElemsPerThread, numElemsPerThread><<<numArrays, numElemsPerArray / numElemsPerThread>>>(d_in, d_out1); 
    gpuErrchk(cudaMemcpy(h_result1, d_out1, N*sizeof(int), cudaMemcpyDeviceToHost));
    
	printf("BlockSortKernel no shared\n\n");
	for (int k = 0; k < numArrays; k++) 
		for (int i = 0; i < numElemsPerArray; i++)
			printf("Array nr. %i; Element nr. %i; Value %i\n", k, i, h_result1[k * numElemsPerArray + i]);
	
	// --- BlockSortKernel with shared
	gpuErrchk(cudaMemcpy(d_in, h_data, N*sizeof(int), cudaMemcpyHostToDevice));
    shared_BlockSortKernel<N / numArrays / numElemsPerThread, numElemsPerThread><<<numArrays, numElemsPerArray / numElemsPerThread>>>(d_in, d_out2); 
    gpuErrchk(cudaMemcpy(h_result2, d_out2, N*sizeof(int), cudaMemcpyDeviceToHost));
    
	printf("\n\nBlockSortKernel with shared\n\n");
	for (int k = 0; k < numArrays; k++) 
		for (int i = 0; i < numElemsPerArray; i++)
			printf("Array nr. %i; Element nr. %i; Value %i\n", k, i, h_result2[k * numElemsPerArray + i]);

	return 0;
}
