#include "Hip_LU_fac.hpp"
#include "roc_kernel.hpp"


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
__global__ void transposeCoalesced(__half *odata, const float *idata, long olda, long ilda)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx_x * TILE_DIM + threadIdx_x;
  int y = blockIdx_y * TILE_DIM + threadIdx_y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx_y+j][threadIdx_x] = idata[(y+j)*ilda + x];

  __syncthreads();

  x = blockIdx_y * TILE_DIM + threadIdx_x;  // transpose block offset
  y = blockIdx_x * TILE_DIM + threadIdx_y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*olda + x] = __float2half(tile[threadIdx_x][threadIdx_y + j]);
}

void half_conversion_right_trans2(float* C, long start_row, long start_col, long b, long nprow, long npcol , __half* rp, long rplda)
{
  // rplda should be passed in as npcol-start_col
  long start_row_add = start_row * b;
  long start_col_add = start_col*b*b*nprow;
  
  dim3 dimGrid(b/TILE_DIM, b*(npcol-start_col)/TILE_DIM, 1); 
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  hipLaunchKernelGGL(transposeCoalesced, dimGrid, dimBlock,0,0,rp, &C[start_row_add + start_col_add], rplda, b*nprow);

  //transposeCoalesced<<<dimGrid, dimBlock>>>(rp, &C[start_row_add + start_col_add], rplda, b*nprow);

}


__global__ void half_conversion_left(float* C, long start_row, long start_col, long b, long nprow,  __half* lp, long lplda)
{
  // lplda should be passed in as nprow-start_row
  int index = blockIdx_x * blockDim_x + threadIdx_x;
  //int index=threadIdx.x;
  long start_row_add = start_row * b;
  long start_col_add = start_col*b*b*nprow;
  for(int i =0; i< b;i++) //column index
  {
    if(index < lplda)
      lp[index + i*lplda] = __float2half(C[start_row_add + start_col_add + index + i*nprow*b]); 
  }
}





