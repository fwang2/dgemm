
CC=clang

## GCC/CrayPE
#CFLAGS = -O3 -fopenmp
# LDFLAGS =

## AMD ROM
## module load rocm
## module load craype-accel-amd-gfx908

CFLAGS = -fopenmp -target x86_64-pc-linux-gnu 		\
			-fopenmp-targets=amdgcn-amd-amdhsa   	\
			-Xopenmp-target=amdgcn-amd-amdhsa    	\
			-march=gfx908

#  OMP_NUM_THREADS=128 ./mt-dgemm 4096


#CFLAGS = -O3 -fopenmp -D USE_BLIS
#LDFLAGS = -lblis-mt


mt-dgemm: mt-dgemm.c
	$(CC) $(CFLAGS) -o mt-dgemm mt-dgemm.c $(LDFLAGS)

clean:
	rm -f mt-dgemm

