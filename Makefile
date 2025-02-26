## default on the system, but writing it here explicity for readability
OPENBLAS_INCLUDE=/home/as1669/local/include
OPENBLAS_LIB=/home/as1669/local/lib

EXECS = pthread_add_fp16 pthread_add_fp32 pthread_adam_fp16 pthread_adam_fp32 pthread_basic_add_fp32 cblas_add omp_add

all: ${EXECS}

## Contain Kernels in AVX512 and specialized for Intel Skylake Server

## THESE KERNELS SHOULD BE GOOD TO GO (thanks o3-mini-high!)

pthread_add_fp16: pthread_add_fp16.c
	gcc -O3 -march=native -mtune=native -pthread -mavx512fp16 -mavx512bf16 -mavx512f  $^ -o $@ 

pthread_add_fp32: pthread_add_fp32.c
	gcc -O3 -march=native -mtune=native -pthread -mavx512f  $^ -o $@

pthread_adam_fp16: pthread_adam_fp16.c
	gcc -O3 -march=native -mtune=native -pthread -lm -mavx512fp16 -mavx512bf16 -mavx512f  $^ -o $@

pthread_adam_fp32: pthread_adam_fp32.c
	gcc -O3 -march=native -mtune=native -pthread -lm -mavx512f  $^ -o $@


## Saturing memory bandwidth but no vector intrinsics

pthread_basic_add_fp32: pthread_basic_add_fp32.c
	gcc -O3 -march=native -mtune=native -pthread -mavx512f $^ -o $@


## Bad versions of add that cannot saturate memory bandwidth


cblas_add: cblas_add.c
	gcc -O3 -march=native -mtune=native -mavx512f -mavx512fp16 -mavx512bf16 $^ -o $@ -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lopenblas

omp_add: omp_add.c
	gcc -O3 -march=native -mtune=native -fopenmp -mavx512f -mavx2 -mf16c $^ -o $@

clean:
	rm -f ${EXECS} *.o

