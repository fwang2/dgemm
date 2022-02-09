#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "hip/hip_runtime.h"
//#include <cublas_v2.h>
//#include "cuda_fp16.h"
#include <chrono>
#include "Hip_LU_fac.hpp" 
#include "roc_kernel.hpp"
//#include <cusolverDn.h>
#include <omp.h>
#include <time.h>
#include <mpi.h>

// Include Hip/Rocm Lib here

using namespace std;

#define MixMM

class timer
{
protected:
    double wtime;

    std::chrono::high_resolution_clock::time_point tStart;
    int isStarted;

public:
    std::string name;
    timer()
    {
        wtime = 0;
        isStarted = 0;
    };
    timer(std::string str)
    {
        wtime = 0;
        isStarted = 0;
        name = str;
    };
    void start()
    {
        isStarted = 1;
        tStart = std::chrono::high_resolution_clock::now();
    }; // start the timer
    void end()
    {
        auto elapsed = std::chrono::high_resolution_clock::now() - tStart;
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        wtime += 1e-6 * (double)microseconds;
        isStarted = 0;

    }; // end the timer
    double getWtime() { return wtime; };
    void reset()
    {
        wtime = 0;
        isStarted = 0;
    }; // resets the timer

    void print()
    {
        std::cout << name;
        printf("\t\t:  \t%.3g \n", wtime);
    }
};

class GPUtimer : public timer
{
public:
    GPUtimer(std::string str) : timer(str)
    {
    }
    void start()
    {
        cudaDeviceSynchronize();
        timer::start();
    }
    void end()
    {
        cudaDeviceSynchronize();
	    timer::end();
    }
};

class counter
{

public:
    double count;
    std::string name;
    counter(std::string str)
    {
        count = 0.0;
        name = str;
    };
    double getCount() { return count; };
    void add(size_t val) { count += (double)val; }
    void reset() { count = 0.0; };
    void print()
    {
        std::cout << name;
        printf("\t\t:  \t%.3g \n", count);
    }
};




const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
       /* case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";*/ 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

// Fill the matrix A with diagonal dominated matrix  CPU
void CPU_fill_rand(float *A, long m, long n, long b) {
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      if(i !=j )
        A[i+j*n] = 1;
      else
        A[i+j*n] = m;    
    }
  }
}
/*
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
}*/

/*const int TILE_DIM = 32;
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

}*/



void print_matrix(float* A, long lda, long size)
{
  long numcolum = size/lda;
  for(long i = 0; i< lda; i++){
    for(long j = 0; j < numcolum; j++)
      printf("%.3f ", A[i+j*lda]);
    printf("\n");
  }
  printf("\n");	
}


void print_matrix(__half* A, long lda, long size)
{
  long numcolum = size/lda;
  for(long i = 0; i< lda; i++){
    for(long j = 0; j < numcolum; j++)
      printf("%.3f ", __half2float(A[i+j*lda]));
    printf("\n");
  }
  printf("\n");	
}

int main(int argc, char ** argv){

  long lda, ldb, ldc;
  long  m = atol(argv[1]);
  long  n = m;
  long  b = atol(argv[2]);
  int bmax = 1024;
  double begin, end;


  cout << "code start"<<std::flush << endl;
  lda = m;
  ldb = b; 
  ldc = m; //Think as nprow*b in distributed setting
  begin = clock();
  __half *h_A = (__half *)malloc((long)m*(long)b *(long) sizeof(__half));
  __half *h_B = (__half *)malloc((long)b*(long)n *(long) sizeof(__half));
  __half *d_A, *d_B;
  float *h_C = (float *)malloc((long)m*(long)n * (long)sizeof(float));
  begin = clock()-begin;
  cout << "done cpu allocation "<<((double)begin)/CLOCKS_PER_SEC<<std::flush << endl;
  
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

   #pragma omp parallel 
   {
     fprintf(stderr, "hello world %d/%d\n", omp_get_thread_num(), omp_get_max_threads());
     
   }

  begin = clock();
  CPU_fill_rand(h_C, (long)m,(long)n,(long)b);
  begin = clock()-begin;
  cout << "done cpu init "<<((double)begin)/CLOCKS_PER_SEC<<std::flush << endl;
  
  begin = clock();
  float *d_C;
  checkCuda(cudaMalloc((void**)&d_A, 1ull*(long)m*(long)b * (long)sizeof(__half)));
  checkCuda(cudaMalloc((void**)&d_B, 1ull*(long)b*(long)n * (long)sizeof(__half)));
  checkCuda(cudaMalloc((void**)&d_C, 1ull*(long)m*(long)n * (long)sizeof(float)));
  checkCuda(cudaMemcpy(d_C,h_C,(long) m * (long)n * (long)sizeof(float),cudaMemcpyHostToDevice));
  begin = clock()-begin;
  cout << "done gpu allocation "<<((double)begin)/CLOCKS_PER_SEC<<std::flush << endl;
  
  long nprow = m/b;
  long npcol = n/b;

  int repeats = 2;
  int verbose = 1;
  //cout << "\ncublasLUfactorization test result:\n"<<std::flush << endl;

  
  timer ti("Input timer");
  timer tt("Total Timer");
  GPUtimer tdiag("getrf Timer");
  GPUtimer tconvert("convert Timer");
  GPUtimer ttrsm("trsm Timer");
  GPUtimer tsGemm("shgemm Timer");


  // Print out info of this run
  if(verbose) 
    cout << "running with" 
	 << " m: " << m
	 << " b: " << b
	 << " repeats: " << repeats
	 << endl;

  cublasStatus_t stat;
  cublasHandle_t handle;
  cublasHandle_t shandle;
  checkCublas(cublasCreate(&handle));
  checkCublas(cublasCreate(&shandle));

  const float alf = 1.0f;
  const float nalf= -1.0f;
  const float bet = 1.0f;
  const float *alpha = &alf;
  const float *beta = &bet;
  

  //printf("Going to main loop\n");
  // Main Loop
  long ki = 0; // Index for traversing by block
  int* inforArry=NULL;
  double st, ed;
  int lwork;
  checkCuda(cudaMalloc((void**)&inforArry, sizeof(int)));
 
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;
  cusolverStatus_t cus_status = CUSOLVER_STATUS_SUCCESS; 
  cudaError_t cudaStat1 = cudaSuccess;
  
  cus_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cus_status);
/*
  cus_status = cusolverDnSgetrf_bufferSize(
      cusolverH,
      b,
      b,
      d_C,
      ldc,
      &lwork);
  assert(CUSOLVER_STATUS_SUCCESS == cus_status);
  float *d_work = NULL;
  cudaStat1 = cudaMalloc((void**)&d_work, 1ull*2*sizeof(float)*lwork);
  assert(cudaSuccess == cudaStat1);*/

//printf("line:allocation %s\n",cudaGetErrorString(cudaGetLastError()));
  begin = clock();
  rocblas_initialize();
  begin = clock()-begin;
 
  cout << "mainLopp, RocBlasInit: "<<((double)begin)/CLOCKS_PER_SEC<<std::flush << endl;
 
  tt.start();

  for( ki=0; ki<nprow;ki++) 
  {
//    printf("Iteration %ld\n",ki);// , ttime: %lf\n", ki, tt.getWtime());
/*   tdiag.print();
    tconvert.print();
    ttrsm.print();
    tsGemm.print();
*/

 
	//Diagonal Sovle: // C is in column major
    float* diag_address= (&d_C[ki*b+ki*nprow*b*b]);
    tdiag.start();
    cus_status = rocsolver_sgetrf_npvt(
			cusolverH, b, b, &d_C[ki*b+ki*nprow*b*b], ldc, inforArry);

/*	cus_status = cusolverDnSgetrf(
            cusolverH,
            b,
            b,
            &d_C[ki*b+ki*nprow*b*b],
            ldc,
            d_work,
            NULL,
            &inforArry);*/
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cus_status);
    assert(cudaSuccess == cudaStat1);
	tdiag.end();

//printf("line:386 %s\n",cudaGetErrorString(cudaGetLastError()));  
    // checkCuda(cudaMemcpy(h_C,d_C, m * n * sizeof(float),cudaMemcpyDeviceToHost));
    /*printf("After Diag\n");
    print_matrix(h_C,m,m*m);*/
	if(ki == nprow-1)
		break;
    //Solve left pannel, A (L21), solution ovewrited in C
    //  L21 * U11 = C21, so matrix U11(solved, stored in C(ki,ki)) is on the right side, with Upper triagular
    //  L21 is size of m-ki*b x b
   
    ttrsm.start();
    stat = cublasStrsm(handle, 
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, 
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                (nprow-ki-1)*b , b , 
                alpha, 
                &d_C[ki*b + ki*nprow*b*b], ldc,  
                &d_C[(ki+1)*b + ki*nprow*b*b], ldc );
	/*stat = rocblas_strsm(handle, 
				rocblas_side_right, rocblas_fill_upper,
				 rocblas_operation_none, rocblas_diagonal_non_unit,
				(nprow-ki-1)*b , b , 
                alpha, 
                &d_C[ki*b + ki*nprow*b*b], ldc,  
                &d_C[(ki+1)*b + ki*nprow*b*b], ldc );*/
 

   checkCublas(stat);


//printf("line:407 %s\n",cudaGetErrorString(cudaGetLastError()));
    //Solve Right pannel, B (U12), solution overwrited in C
    //  L11 * U12 = C12, so matrix L11(solved, stored in C(ki,ki)) is on the left side, with lower triagular 
    stat = cublasStrsm(handle, 
               CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
               CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 
               b , (npcol-ki-1)*b , 
               alpha, 
               &d_C[ki*b + ki*nprow*b*b], ldc,  
               &d_C[ ki*b + (ki+1)*nprow*b*b], ldc );

	checkCublas(stat);


	//printf("line:417 %s\n",cudaGetErrorString(cudaGetLastError()));
	ttrsm.end();



    //checkCuda(cudaMemcpy(h_C,d_C, m * n * sizeof(float),cudaMemcpyDeviceToHost));
    /*printf("After Panels Solves\n");
    print_matrix(h_C,m,m*m);*/

    //Conversion, here we can play the trick to try tanspose version, and other formating techique.
    //  Since we have to copy it out to do coversion or sending, we change the formating here (Tranpose, row major, tiling, and others)
    //  In addition, we can play with the thread blocks
	tconvert.start();


    /*dim3 block_dims( ceil((float)m/1024), 1, 1);
    dim3 thread_dims(1024, 1, 1);
   	hipLaunchKernelGGL( half_conversion_left,block_dims, thread_dims,0,0,d_C, ki+1, ki , b, nprow, d_A, b*(nprow-(ki+1)));
	*/half_conversion_right_trans2(d_C, ki, ki+1, b, nprow, npcol ,d_B , b*(npcol-(ki+1)));
    tconvert.end();
//printf("line:431 %s\n",cudaGetErrorString(cudaGetLastError()));   
    // U12 is now stored in B as U12' (transpose)

	


//printf("line:436 %s\n",cudaGetErrorString(cudaGetLastError()));
    //checkCuda(cudaMemcpy(h_A,d_A, m * b * sizeof(__half),cudaMemcpyDeviceToHost));
    /*printf("A\n");
    print_matrix(h_A,(nprow-ki-1)*b, b*b*(nprow-ki-1));
    checkCuda(cudaMemcpy(h_B,d_B, b * n * sizeof(__half),cudaMemcpyDeviceToHost));
    printf("B\n");
    print_matrix(h_B,(nprow-ki-1)*b,(nprow-ki-1)*b*b);
    */


    //Update rest C(ki+1:m, ki+1:n)
    // A22 – L21 * U12

    tsGemm.start();
    /*stat = cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, (nprow-(ki+1))*b, (npcol-(ki+1))*b, b, &nalf, 
							(void*)d_A, CUDA_R_16F ,b*(nprow-(ki+1)), 
							(void*)d_B, CUDA_R_16F ,b*(npcol-(ki+1)), 
  							beta, 
                            &d_C[(ki+1)*b+(ki+1)*nprow*b*b], CUDA_R_32F ,ldc);
	*/
	stat = rocblas_gemm_ex(shandle, CUBLAS_OP_N, CUBLAS_OP_T, 
		(nprow-(ki+1))*b, (npcol-(ki+1))*b, b, &nalf,
		(void*)d_A, CUDA_R_16F ,b*(nprow-(ki+1)), 
		(void*)d_B, CUDA_R_16F ,b*(npcol-(ki+1)),	
		beta, 
		&d_C[(ki+1)*b+(ki+1)*nprow*b*b], CUDA_R_32F ,ldc,
		&d_C[(ki+1)*b+(ki+1)*nprow*b*b], CUDA_R_32F ,ldc,
		CUDA_R_32F, rocblas_gemm_algo_standard, 0,0);
 	checkCublas(stat);
    
    tsGemm.end();



	checkCublas(stat);
/*
    checkCuda(cudaMemcpy(h_C,d_C, m * n * sizeof(float),cudaMemcpyDeviceToHost));
    printf("After Shgemm\n");
    print_matrix(h_C,m,m*m); 
printf("line:459 %s\n",cudaGetErrorString(cudaGetLastError()));
*/

  }
//  	checkCuda(cudaMemcpy(h_C,d_C, m * n * sizeof(float),cudaMemcpyDeviceToHost));
//    printf("Final\n");
//    print_matrix(h_C,m,m*m); 
//	printf("line:459 %s\n",cudaGetErrorString(cudaGetLastError()));

	//cudaEventRecord(stop,0);
  	//cudaEventSynchronize(stop);
  if(stat != CUBLAS_STATUS_SUCCESS){
    cerr << "cublasSgemmBatched failed" << endl;
    exit(1);
  }
  assert(!cudaGetLastError());
  //printf("line:470 %s\n",cudaGetErrorString(cudaGetLastError()));

  cout << "\nFinal"<<std::flush << endl;
  
  tt.end();
  tt.print();
  tdiag.print();
  tconvert.print();
  ttrsm.print();
  tsGemm.print();



  // long ops = 2*(long)m*(long)n*(long)k;
  // double opss =  (double)ops/(double)(sum/repeats)/(double)1000000000;  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  //cudaFree(d_work);
  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
  MPI_Finalize();
  return 0;
}
