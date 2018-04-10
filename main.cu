#include <iostream>
#include <iomanip>
#include <ctime>
#include <cfloat>
#include <pthread.h>
#include <cuda.h>
#include <stdio.h>

using namespace std;

typedef double (*integrable)(double);

__host__ __device__ double parabola(double x)
{ return x * x; }

typedef struct integration_args_tag
{
    unsigned long long bstep;
    unsigned long long estep;
    double h;
    double a;
    double b;
    integrable f;
    double result;
} integration_args;



__global__ void integrate(double a, double b, integrable f, unsigned long long steps, double *result)
{
	extern __shared__ double pre_sums[];
	const unsigned long long thread_count = blockDim.x;
	unsigned long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long steps_per_thread = steps / thread_count;
    unsigned long long bstep = thread_id *  steps_per_thread;
    unsigned long long estep = bstep + steps_per_thread;
    pre_sums[thread_id] = 0;
    double h = (b - a) / (steps * 1.0);

    for (unsigned long long i = bstep; i < estep; i++)
    {
        double x = a + i * h;
        pre_sums[thread_id] += h * f(x);
    }

    __syncthreads();
    if(thread_id == 0)
    {
    	*result = 0;
    	for (unsigned long long i = 0; i < thread_count; i++) *result += pre_sums[i];
    }

}

__device__ integrable d_f = parabola;


int main(int argc, char *argv[])
{
    const unsigned int thread_count = 512;
    const unsigned long long steps = 1000000000;
    double a = 0;
    double b = 1;
    unsigned long long steps_per_thread = steps / thread_count;
    double *d_result;
    cudaMalloc(&d_result, sizeof(double));
    integrable h_fun;
    cudaMemcpyFromSymbol(&h_fun, d_f, sizeof(integrable));
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin);
    int total_blocks = 4;
    dim3 grid;
    grid.x = total_blocks;
    dim3 block_geom;
    block_geom.x = thread_count / total_blocks;
    integrate<<<grid, block_geom,  sizeof(double) * thread_count>>>(a, b, h_fun, steps, d_result);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    double sum;
    cudaMemcpy(&sum, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cout << fixed << setprecision(DBL_DIG) << sum << endl;
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cout << fixed << setprecision(DBL_DIG) << time / 1000  << " seconds." << endl;
    cudaFree(d_result);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return 0;
}
