#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

int main() {
    uint64_t n = 485000000;
    
    uint64_t dtype_size = 1;

    uint8_t * a = (uint8_t *)malloc(n * sizeof(int));
    uint8_t * b= (uint8_t*)malloc(n * sizeof(int));
    uint8_t * c = (uint8_t*)malloc(n * sizeof(int));

    for (uint64_t i = 0; i < n; i++) {
        a[i] = rand() & 0xFF;
	b[i] = rand() & 0xFF;
    }

    struct timespec start_time, stop_time;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    #pragma omp parallel for
    for (uint64_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &stop_time);

    uint64_t start_timestamp = start_time.tv_sec * 1e9 + start_time.tv_nsec;
    uint64_t stop_timestamp = stop_time.tv_sec * 1e9 + stop_time.tv_nsec;

    uint64_t elapsed_ns = stop_timestamp - start_timestamp;

    double elapsed_ms = (double) elapsed_ns / 1e6;
	
    uint64_t total_bytes = 3 * n * dtype_size;
	
    double mem_bw_util = ((double) total_bytes / 1e9) / ((double) elapsed_ns / 1e9);

    printf("\nElapsed Time (ms): %.3f\nMem BW Util (GB / sec): %.3f\n", elapsed_ms, mem_bw_util);


    free(a);
    free(b);
    free(c);

    return 0;
}
