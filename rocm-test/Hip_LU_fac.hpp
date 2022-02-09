#ifndef __AMD_CUBLASMAP__
#define __AMD_CUBLASMAP__

#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "rocblas.h"
#include "rocsolver.h"

#define cublasStatus_t rocblas_status
#define CUBLAS_STATUS_SUCCESS rocblas_status_success
#define CUBLAS_STATUS_NOT_INITIALIZED rocblas_status_invalid_handle
#define CUBLAS_STATUS_ALLOC_FAILED rocblas_status_memory_error
#define CUBLAS_STATUS_INVALID_VALUE rocblas_status_invalid_value
#define CUBLAS_STATUS_ARCH_MISMATCH rocblas_status_internal_error
#define CUBLAS_STATUS_MAPPING_ERROR rocblas_status_internal_error
#define CUBLAS_STATUS_EXECUTION_FAILED rocblas_status_internal_error
#define CUBLAS_STATUS_INTERNAL_ERROR rocblas_status_internal_error

#define cublasStrsm rocblas_strsm
#define CUBLAS_SIDE_RIGHT rocblas_side_right
#define CUBLAS_FILL_MODE_UPPER rocblas_fill_upper
#define CUBLAS_OP_N rocblas_operation_none
#define CUBLAS_OP_T rocblas_operation_transpose
#define CUBLAS_DIAG_NON_UNIT rocblas_diagonal_non_unit
#define CUBLAS_SIDE_LEFT rocblas_side_left
#define CUBLAS_FILL_MODE_LOWER rocblas_fill_lower
#define CUBLAS_DIAG_UNIT rocblas_diagonal_unit
#define CUDA_R_16F rocblas_datatype_f16_r
#define CUDA_R_32F rocblas_datatype_f32_r


#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost 
#define cudaGetLastError hipGetLastError
/*
#define threadIdx_x hipThreadIdx_x
#define threadIdx_y hipThreadIdx_y
#define blockIdx_x hipBlockIdx_x
#define blockIdx_y hipBlockIdx_y
#define blockDim_x hipBlockDim_x
#define blockDim_y hipBlockDim_y
#define gridDim_x hipGridDim_x
#define gridDim_y hipGridDim_y*/
#define cudaFree hipFree
#define cudaStream_t hipStream_t
//#define __half 

#define cudaMalloc hipMalloc
#define cudaMallocHost hipMallocHost

#define cublasHandle_t rocblas_handle
#define cublasCreate rocblas_create_handle
#define cublasSetStream rocblas_set_stream
#define cusolverDnHandle_t rocblas_handle
#define cusolverStatus_t rocblas_status
#define CUSOLVER_STATUS_SUCCESS rocblas_status_success
#define cusolverDnCreate rocblas_create_handle



#endif
