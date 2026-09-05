#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// Each thread processes one complete binary-tree node at the current
// level.  Separate launches provide the global synchronization required
// between levels of the Blelloch scan.
__global__ void upsweep_kernel(int* data, int n, int stride) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    const int nodes = n / (2 * stride);
    if (node >= nodes) return;

    // Validate node before forming this product.  At the top of a large
    // tree, inactive threads in a 256-thread block would otherwise make
    // node * 2 * stride overflow a signed int.
    const int right = (node + 1) * (2 * stride) - 1;
    data[right] += data[right - stride];
}

__global__ void set_last_to_zero_kernel(int* data, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        data[n - 1] = 0;
    }
}

__global__ void downsweep_kernel(int* data, int n, int stride) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    const int nodes = n / (2 * stride);
    if (node >= nodes) return;

    const int right = (node + 1) * (2 * stride) - 1;
    int left = right - stride;
    int saved_left = data[left];
    data[left] = data[right];
    data[right] += saved_left;
}


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    // cudaScan() has already copied input into result, so scanning result
    // in place avoids another global-memory copy.  The padded portion may
    // contain arbitrary values; it can only affect padded output entries.
    (void)input;
    const int scan_length = nextPow2(N);

    for (int stride = 1; stride < scan_length; stride *= 2) {
        const int nodes = scan_length / (2 * stride);
        const int blocks = (nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, scan_length, stride);
    }

    set_last_to_zero_kernel<<<1, 1>>>(result, scan_length);

    for (int stride = scan_length / 2; stride >= 1; stride /= 2) {
        const int nodes = scan_length / (2 * stride);
        const int blocks = (nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, scan_length, stride);
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_result);

    double overallDuration = endTime - startTime;
    return overallDuration;
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void
find_repeats_kernel(int* device_input, int N, int* result) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if (index < N - 1) {
        if (device_input[index] == device_input[index + 1]) {
            result[index] = 1;
        } else {
            result[index] = 0;
        }
    }
}

__global__ void
write_repeats_kernel(int* flags, int* scan, int N, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N && flags[index] == 1) {
        int output_index = scan[index];
        result[output_index] = index;
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    if (length <= 1) {
        return 0;
    }

    // exclusive_scan processes nextPow2(length) elements.  Allocate and
    // zero-pad both temporary arrays so that those extra elements neither
    // access invalid memory nor contribute to the scan.
    const int scan_length = nextPow2(length);

    int *device_flags;
    int *device_scan;

    cudaMalloc((void **)&device_flags, sizeof(int) * scan_length);
    cudaMalloc((void **)&device_scan, sizeof(int) * scan_length);

    const int blocks = (scan_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaMemset(device_flags, 0, scan_length * sizeof(int));
    find_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_input, length, device_flags);

    cudaMemcpy(device_scan, device_flags, scan_length * sizeof(int), cudaMemcpyDeviceToDevice);
    exclusive_scan(device_flags, length, device_scan);

    write_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_flags, device_scan, length, device_output);

    // flags[length - 1] is always zero, since it has no right neighbour.
    // Thus its exclusive-scan value is exactly the number of repeats.
    int count;
    cudaMemcpy(&count, device_scan + length - 1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_flags);
    cudaFree(device_scan);
    return count;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);

    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
