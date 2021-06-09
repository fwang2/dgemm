========================================================================================
Crossroads/NERSC-9 DGEMM Compute Benchmark
========================================================================================
Benchmark Version: 1.0.0

========================================================================================
Benchmark Description:
========================================================================================
The Crossroads/NERSC-9 Memory Bandwidth benchmark is a simple single-node multi-threaded 
dense-matrix multiply benchmark. The code is designed to demonstrate high floating-point 
compute rates on a machine under sustained computation. The Offeror is expected to run 
this benchmark to report compute performance and not to report peak hardware performance.

========================================================================================
Permitted Modifications:
========================================================================================
Offerors are permitted to modify the benchmark in the following ways:

OpenMP Pragmas - the vendor may modify the OpenMP pragmas in the benchmark as needed 
provided the resulting program remains a standards compliant language and OpenMP 
program (compliant to the language/OpenMP specification proposed by the Offeror in 
their response). Any modifications made to the benchmark should be included in the 
Offeror's response.

Call to Optimized Libraries - the Offeror may replace the core matrix multiplication 
call by a call to a vendor optimized library (such as BLAS, MKL, CuBLAS etc). Any 
modification made to the benchmark should be included in the OfferorÕs response including 
a complete copy of the modified source code. Any libraries used to modify the benchmark 
must be included in the OfferorÕs system proposal.

Problem Size - the Offeror may modify the input configuration to the benchmark 
(by supplying a command line parameter, not modifying the code) to increase the problem 
size. The problem size must meet the requirement N>=128. The Offeror must include the 
value of N used in their response.

Problem Repetitions - the Offeror may modify the repeated runs ("repeats") of the 
benchmark provided the number of repeats is at >= 30. The Offeror must include the 
number of repetitions executed in their response.

Matrix Padding - padding of input matrices may be performed to improve performance. 
The Offeror must provide the modifications to the source code to achieve padding in 
their response.

========================================================================================
Run Rules:
========================================================================================
The Offeror may utilize any number of threads, affinity and memory binding options for 
execution of the benchmark provided: (1) details of all command line parameters, 
environment variables and binding tools are included in the response; (2) details of 
all the compute cores/threads/units utilized and their arrangement within the node must 
be described.

The Offeror is expected to provide the GFLOP/s rate as reported by the benchmark for each 
type of compute core/unit used in the proposed design. Different values of N and repeats 
are permitted to be used for each type of compute core/unit but these must be reported in 
the response.

The Offeror is expected to describe the percentage of theoretical double-precision 
(64-bit) computation peak which the benchmark GFLOP/s rate achieves for each type of 
compute core/unit in the response and describe how this is calculated.

The type of matrix elements is currently set to double precision (64-bit). This data 
type must not be changed.

The benchmark matrix self-check must correctly pass for the benchmark to be considered 
a valid run. The current benchmark code includes a source-code check with appropriate 
print outs. This check must not be modified.

========================================================================================
How to Compile, Run and Verify:
========================================================================================
To build simply type modify the file Makefile for your compiler and type make. To run, 
execute the file mt-dgemm. Mt-dgemm does self verification.

$ make
<lots of make output>
$ export OMP_NUM_THREADS=12
$ ./mt-dgemm
<mt-dgemm output>

The size of the matrix can be changed on the command line, e.g. 

# ./mt-dgemm 4096

will execute using 4096x4096 block matrices. 

========================================================================================
How to report
========================================================================================
The primary FOM is "GFLOP/s rate". Report all data printed to stdout.

