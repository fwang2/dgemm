//#include "rocblas.h"
#include "hip/hip_runtime.h"
//#include "rocsolver.h"
#include "hip/hip_fp16.h"


#define threadIdx_x hipThreadIdx_x
#define threadIdx_y hipThreadIdx_y
#define blockIdx_x hipBlockIdx_x
#define blockIdx_y hipBlockIdx_y
#define blockDim_x hipBlockDim_x
#define blockDim_y hipBlockDim_y
#define gridDim_x hipGridDim_x
#define gridDim_y hipGridDim_y


__global__ void transposeCoalesced(__half *odata, const float *idata, long olda, long ilda);
void half_conversion_right_trans2(float* C, long start_row, long start_col, long b, long nprow, long npcol , __half* rp, long rplda);

__global__ void half_conversion_left(float* C, long start_row, long start_col, long b, long nprow,  __half* lp, long lplda);




