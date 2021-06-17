# dgemm



## using ROCM 

Load up the AMD compiler on Spock:

```
module load rocm
module load craype-accel-amd-gfx908
```
**Note** both modules must be loaded for GPU offload to work.


The compiler flags for enabling GPU offload for OpenMP is the following:

```
CFLAGS = -fopenmp -target x86_64-pc-linux-gnu 		\
			-fopenmp-targets=amdgcn-amd-amdhsa   	\
			-Xopenmp-target=amdgcn-amd-amdhsa    	\
			-march=gfx908
```


To verify:



## case 1

```
./mt-dgemm 10000

