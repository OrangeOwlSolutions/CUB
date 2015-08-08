#define CUB_STDERR

#include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_reduce.cuh>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

using namespace cub;

/********/
/* MAIN */
/********/
int main() {

    const int N = 8388608;

    gpuErrchk(cudaFree(0));

    float *h_data		= (float *)malloc(N * sizeof(float));
	float h_result = 0.f;

	for (int i=0; i<N; i++) {
		h_data[i] = 3.f;
		h_result = h_result + h_data[i];
	}

	TimingGPU timerGPU;

	float *d_data;			gpuErrchk(cudaMalloc((void**)&d_data, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    /**********/
    /* THRUST */
    /**********/
    timerGPU.StartCounter();
    thrust::device_ptr<float> wrapped_ptr = thrust::device_pointer_cast(d_data);
	float h_result1 = thrust::reduce(wrapped_ptr, wrapped_ptr + N);
	printf("Timing for Thrust = %f\n", timerGPU.GetCounter());

    /*******/
    /* CUB */
    /*******/
    timerGPU.StartCounter();
    float			*h_result2 = (float *)malloc(sizeof(float));
    float			*d_result2;	gpuErrchk(cudaMalloc((void**)&d_result2, sizeof(float)));
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

	DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_result2, N);
	gpuErrchk(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes));
	DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_result2, N);

	gpuErrchk(cudaMemcpy(h_result2, d_result2, sizeof(float), cudaMemcpyDeviceToHost));

	printf("Timing for CUB = %f\n", timerGPU.GetCounter());

	printf("Results:\n");
	printf("Exact: %f\n", h_result);
	printf("Thrust: %f\n", h_result1);
	printf("CUB: %f\n", h_result2[0]);

}
