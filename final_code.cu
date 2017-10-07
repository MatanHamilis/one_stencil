/////////////////////////////////////////////////////////

// Computes the 1-stencil using GPUs.
// We don't check for error here for brevity.
// In your implementation - you must do it!

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

#ifndef k
#define k 3
#endif
#ifndef OUTPUT_PER_THREAD
#define OUTPUT_PER_THREAD 1
#endif
#define LOCAL_REGISTER_SIZE ((1+OUTPUT_PER_THREAD) > (k+31)/32 ? (1+OUTPUT_PER_THREAD) : (k+31)/32)
#ifndef TEST_TIMES
#define TEST_TIMES 5
#endif
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <cuda.h>
float host_k_stencil (int *A, int *B, int sizeOfA, int withRc);
__global__ void k_stencil (int *A, int *B, int sizeOfA);
__global__ void k_stencil_with_rc (int *A, int *B, int sizeOfA);

float host_one_stencil (int *A, int *B, int sizeOfA, int withRc);
bool ans_check(int *A, int* B, int sizeOfA, int t);
__global__ void one_stencil (int *A, int *B, int sizeOfA);
__global__ void one_stencil_with_rc (int *A, int *B, int sizeOfA);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char*file, int line, bool abort=true)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void fill_array(int *arr, int size)
{
	for (int i = 0 ; i < size ; ++i)
	{
		arr[i] = i%17;
	}
}
int main(int argc, char** argv)
{
	int withRc = 0;
	if (argc != 3)
	{
		printf("Usage: %s [RC/NO_RC] [TEST_SIZE > 1000]\n", argv[0]);
		exit(-1);
	}

	if (0 == strcmp("RC", argv[1]))
	{
		withRc = 1;	
	}

	int test_size = atoi(argv[2]);

	int *A = new int[test_size];
	int *B = new int[test_size - 2];

	fill_array(A, test_size);
	fill_array(B, test_size);
	float sum = 0;
	float min = 0;
	float max = 0;
	float current = 0;
	for (int i = 0 ; i < TEST_TIMES  ; ++i)
	{	
		current = host_k_stencil(A, B, test_size, withRc);
		if (i == 0)
		{
			min = current;
			max = current;
		}

		//printf("%f\n", current);
		sum += current;

		if (min > current)
		{
			min = current;
		}

		if (max < current)
		{
			max = current;
		}

#ifdef TEST_MODE
		if (!ans_check(A, B, test_size, k))
		{
			printf("TEST FAILED!\n");
			break;
		}
#endif
	}
	sum = sum - min - max;
	delete[] A;
	delete[] B;
	printf("%f\n", sum/(TEST_TIMES-2));
	return 0;
		
}

bool ans_check(int *A, int* B, int sizeOfA, int t)
{
	for (int i = 0 ; i < sizeOfA - t - 1 ; ++i)
	{
		int sum = 0;
		for (int j = 0 ; j < t + 1 ; ++j)
		{
			sum += A[i + j];
		}
		if (B[i] != sum)
		{
			return 0;
		}
	}
	return 1;
}
float host_one_stencil (int *A, int *B, int sizeOfA, bool withRc)
{
	// Declare device pointer.
	int *d_A = NULL;
	int *d_B = NULL;
	
	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;

	// Allocate memory on device.
	gpuErrchk(cudaMalloc((void**)(&d_A), sizeOfA * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)(&d_B), (sizeOfA - 2) * sizeof(int)));
	
	// Copy input to device
	cudaMemcpy(d_A, A, sizeOfA * sizeof(int), cudaMemcpyHostToDevice);
	
	// We round up the result of (sizeOfA / BLOCK_SIZE).
	int threadBlocksNum = (sizeOfA + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (withRc)
	{
		cudaEventRecord(start);
		one_stencil_with_rc<<< threadBlocksNum , BLOCK_SIZE >>>(d_A, d_B, sizeOfA);
		cudaEventRecord(stop);
	}
	else
	{
		int neededShmem = (BLOCK_SIZE + 2) * sizeof(int);
		cudaEventRecord(start);
		one_stencil<<< threadBlocksNum , BLOCK_SIZE, neededShmem >>>(d_A, d_B, sizeOfA);	
		cudaEventRecord(stop);
	}
	
	
	cudaDeviceSynchronize();
	gpuErrchk(cudaPeekAtLastError());
	// Read output from device
	cudaMemcpy(B, d_B, (sizeOfA - 2) * sizeof(int), cudaMemcpyDeviceToHost);
	
	// Measure test results
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);	

	// Free memory.
	cudaFree(d_A);
	cudaFree(d_B);
	return ms;
	
}
// Computes the 1-stencil of A using shared memory
// Writes output into B.

__global__ void one_stencil (int *A, int *B, int sizeOfA)
{
	extern __shared__ int s[];
	// Id of thread in the block.
	int localId = threadIdx.x; 
	
	// The first index of output element computed by this block.
	int startOfBlock = blockIdx.x * blockDim.x; 
	
	// The Id of the thread in the scope of the grid.
	int globalId = localId + startOfBlock;
	
	if (globalId >= sizeOfA)
		return;
	
	// Fetching into shared memory.
	s[localId] = A[globalId];
	if (localId < 2 && blockDim.x + globalId < sizeOfA)
	{
		s[blockDim.x + localId] =  A[blockDim.x + globalId];
	}
	
	// We must sync before reading from shared memory.
	__syncthreads();
	
	// Each thread computes a single output.
	if (globalId < sizeOfA - 2)
		B[globalId] = s[localId] + s[localId + 1] + s[localId + 2];
}


// Computes the 1-stencil of A with register cache.
// Writes output into B.
__global__ void one_stencil_with_rc (int *A, int *B, int sizeOfA)
{
	// Declaring local register cache.
	int rc[2];

	// Id of thread in the warp.
	int localId = threadIdx.x % WARP_SIZE; 
	
	// The first index of output element computed by this warp.
	int startOfWarp = blockIdx.x * blockDim.x + WARP_SIZE*(threadIdx.x / WARP_SIZE); 
	
	// The Id of the thread in the scope of the grid.
	int globalId = localId + startOfWarp;
	
	if (globalId >= sizeOfA)
		return;
	
	// Fetching into shared memory.
	rc[0] = A[globalId];
	if (localId < 2 && WARP_SIZE + globalId < sizeOfA)
	{
		rc[1] =  A[WARP_SIZE + globalId];
	}
	
	// Each thread computes a single output.
	int ac = 0;
	int toShare = rc[0];
	for (int i = 0 ; i < 3 ; ++i)
	{
		// Threads decide what value will be published in the following access.
		if (localId < i)
			toShare = rc[1];

		// Accessing register cache.
		unsigned mask = __activemask();
		ac += __shfl_sync(mask, toShare, (localId + i) % WARP_SIZE);
	}

	if (globalId < sizeOfA - 2)
		B[globalId] = ac;
}

float host_k_stencil (int *A, int *B, int sizeOfA, int withRc)
{
	// Declare device pointer.
	int *d_A = NULL;
	int *d_B = NULL;
	
	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;

	// Allocate memory on device.
	gpuErrchk(cudaMalloc((void**)(&d_A), sizeOfA * sizeof(int) + WARP_SIZE*(OUTPUT_PER_THREAD+1)));
	gpuErrchk(cudaMalloc((void**)(&d_B), (sizeOfA ) * sizeof(int) + WARP_SIZE*OUTPUT_PER_THREAD));
	
	// Copy input to device
	cudaMemcpy(d_A, A, sizeOfA * sizeof(int), cudaMemcpyHostToDevice);
	
	// We round up the result of (sizeOfA / BLOCK_SIZE).
	if (withRc == 1)
	{
		int threadBlocksNum = (sizeOfA + OUTPUT_PER_THREAD*BLOCK_SIZE - 1) / (OUTPUT_PER_THREAD*BLOCK_SIZE);
		cudaEventRecord(start);
		k_stencil_with_rc<<< threadBlocksNum , BLOCK_SIZE >>>(d_A, d_B, sizeOfA);
		cudaEventRecord(stop);
	}
	else
	{
		int threadBlocksNum = (sizeOfA + BLOCK_SIZE*OUTPUT_PER_THREAD - 1) / (BLOCK_SIZE*OUTPUT_PER_THREAD);
		int neededShmem = (BLOCK_SIZE*OUTPUT_PER_THREAD+ k) * sizeof(int);
		cudaEventRecord(start);
		k_stencil<<< threadBlocksNum , BLOCK_SIZE, neededShmem >>>(d_A, d_B, sizeOfA);	
		cudaEventRecord(stop);
	}
	
	
	cudaDeviceSynchronize();
	gpuErrchk(cudaPeekAtLastError());
	// Read output from device
	cudaMemcpy(B, d_B, (sizeOfA - k) * sizeof(int), cudaMemcpyDeviceToHost);
	
	// Measure test results
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);	

	// Free memory.
	cudaFree(d_A);
	cudaFree(d_B);
	return ms;
	
}

__global__ void k_stencil (int *A, int *B, int sizeOfA)
{
	extern __shared__ int s[];
	// Id of thread in the block.
	int localId = threadIdx.x; 
	
	// The first index of output element computed by this block.
	int startOfBlock = blockIdx.x * blockDim.x * OUTPUT_PER_THREAD;
	
	// The Id of the thread in the scope of the grid.
	int globalId = localId + startOfBlock;
	
	if (globalId >= sizeOfA)
		return;
	
	// Fetching into shared memory.
	for (int i = 0 ; i < OUTPUT_PER_THREAD ; ++i)
	{
		if (globalId + i*BLOCK_SIZE < sizeOfA)
		{
			s[localId + i*BLOCK_SIZE] = A[globalId + i*BLOCK_SIZE];
		}
	}

	if (localId < k && blockDim.x*OUTPUT_PER_THREAD + globalId < sizeOfA)
	{
		s[localId + blockDim.x*OUTPUT_PER_THREAD] =  A[blockDim.x*OUTPUT_PER_THREAD + globalId];
	}
	
	// We must sync before reading from shared memory.
	__syncthreads();
	
	int sum = 0;
	for (int j = 0 ; j < OUTPUT_PER_THREAD ; ++j)
	{
		sum = 0;
		if (globalId + j*BLOCK_SIZE >= sizeOfA - k)
			return;
		for (int i = 0 ; i < k + 1 ; ++i)
		{
			sum += s[localId + j*BLOCK_SIZE + i]; 
		}
		B[globalId + BLOCK_SIZE*j] = sum ;
	}
}


// Computes the 1-stencil of A with register cache.
// Writes output into B.
__global__ void k_stencil_with_rc (int *A, int *B, int sizeOfA)
{
	// Declaring local register cache.
	int rc[LOCAL_REGISTER_SIZE];

	// Id of thread in the warp.
	int localId = threadIdx.x % WARP_SIZE; 
	
	// The first index of output element computed by this warp.
	int startOfWarp = (blockIdx.x * blockDim.x + WARP_SIZE*(threadIdx.x / WARP_SIZE))*OUTPUT_PER_THREAD; 
	
	// The Id of the thread in the scope of the grid.
	int globalId = localId + startOfWarp;
	
	if (globalId >= sizeOfA)
		return;
	
	// Fetching into shared memory.

#pragma unroll
	for (int i = 0 ; i < OUTPUT_PER_THREAD ; ++i)
	{
		rc[i] = A[(int)(globalId + WARP_SIZE*i)];
	}

	rc[LOCAL_REGISTER_SIZE - 1] =  A[OUTPUT_PER_THREAD*WARP_SIZE + globalId];
	// Each thread computes a single output.

#pragma unroll
	for (int j = 0 ; j < OUTPUT_PER_THREAD ; ++j)
	{
		
		int toShare = rc[j];
		int ac = 0;
#pragma unroll
		for (int i = 0 ; i < k + 1 ; ++i)
		{
			// Threads decide what value will be published in the following access.
			toShare += (i==localId)*(rc[j+1] - rc[j]);
			// Accessing register cache.
			unsigned mask = __activemask();
			ac += __shfl_sync(mask, toShare, (localId + i) & (WARP_SIZE - 1));
		}

			B[globalId + j*WARP_SIZE] = ac ;
			
	}
}
