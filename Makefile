
CC=cc
CFLAGS = -O3 -fopenmp -D USE_BLIS
LDFLAGS = -lblis-mt

mt-dgemm: mt-dgemm.c
	$(CC) $(CFLAGS) -o mt-dgemm mt-dgemm.c $(LDFLAGS)

clean:
	rm -f mt-dgemm

