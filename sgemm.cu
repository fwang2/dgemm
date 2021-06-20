// nvcc sgemm.cu -lcublas -arch=sm_70 -o sgemm

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

const char *cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

inline cublasStatus_t checkCublas(cublasStatus_t result)
{
    if (result != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
        assert(result == CUBLAS_STATUS_SUCCESS);
    }
    return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A)
{
    for (int i = 0; i < nr_rows_A * nr_cols_A; i++)
    {
        A[i] = (float)rand() / (float)(RAND_MAX);
    }
}
int main(int argc, char **argv)
{

    int min_mkn = 1024;
    int max_mkn = 4096*8;
    int repeats = 2;

    cout << "running with"
         << " min_mkn: [" << min_mkn << "]"
         << " max_mkn: [" << max_mkn << "]"
         << " repeats: " << repeats
         << endl;

    cublasStatus_t stat;
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Allocate 3 arrays on CPU

    float *h_A = (float *)malloc(max_mkn * max_mkn * sizeof(float));
    float *h_B = (float *)malloc(max_mkn * max_mkn * sizeof(float));
    float *h_C = (float *)malloc(max_mkn * max_mkn * sizeof(float));

    CPU_fill_rand(h_A, max_mkn, max_mkn);
    CPU_fill_rand(h_B, max_mkn, max_mkn);
    CPU_fill_rand(h_C, max_mkn, max_mkn);

    __half *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged((void **)&d_A, max_mkn * max_mkn * sizeof(__half)));
    checkCuda(cudaMallocManaged((void **)&d_B, max_mkn * max_mkn * sizeof(__half)));
    checkCuda(cudaMallocManaged((void **)&d_C, max_mkn * max_mkn * sizeof(float)));

    // TODO: just initialize on GPU
    for (int i = 0; i < max_mkn * max_mkn; i++)
    {
        d_A[i] = __float2half(h_A[i]);
        d_B[i] = __float2half(h_B[i]);
    }

    checkCuda(cudaMemcpy(d_A, h_A, max_mkn * max_mkn * sizeof(__half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, max_mkn * max_mkn * sizeof(__half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C, h_C, max_mkn * max_mkn * sizeof(float), cudaMemcpyHostToDevice));

    cout << "Transferred A, B, C matrix from host to device" << endl;

    int lda, ldb, ldc, m, n, k;
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double totalSum = 0.0;
    for (int size = min_mkn; size <= max_mkn; size = size + 1024) {
        double sum = 0.0;        
        for(int rep = 0; rep < repeats; rep++) {
            cudaEventRecord(start, 0);
            m = n = k = size;
            lda = m;
            ldb = k;
            ldc = m;
            // NT is the best
            stat = cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha,
                               (void *)d_A, CUDA_R_16F, lda,
                               (void *)d_B, CUDA_R_16F, ldb, beta,
                               (void *)d_C, CUDA_R_32F, ldc);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            if (stat != CUBLAS_STATUS_SUCCESS)
            {
                cerr << "cublasSgemm failed" << endl;
                exit(1);
            }
            assert(!cudaGetLastError());

            float elapsed; // ms returned
            cudaEventElapsedTime(&elapsed, start, stop);
            elapsed /= 1000.0f;
            sum += elapsed; // seconds
            totalSum += sum;
        }
        long ops = 2 * (long)m * (long)n * (long)k;
        double opss = (double)ops / (double)(sum / repeats) / (double)1000000000000;
        cout << fixed << setprecision(4) << "float_Mix: m, n, k = ["        
            << size << "]" << ", Average: " << sum/repeats << " secs, "
            << "TFlops [" << opss << "], Total:" << totalSum/repeats << " secs" << endl;

    }


    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}
