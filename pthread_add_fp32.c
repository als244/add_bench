#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

// Timing helper: returns current time in milliseconds.
double get_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}


// Structure to hold thread arguments.
typedef struct {
    long start;      // starting index (inclusive)
    long end;        // ending index (exclusive)
    float *x;
    float *y;
    float x_scale;
} thread_args;

void *thread_func_fp32(void *arg) {
    thread_args *args = (thread_args *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 16;  // 16 FP32 elements per 512-bit vector.
    long limit = start + (len / chunk) * chunk;
    
    // Broadcast the scalar factor into a 512-bit vector.
    __m512 factor_vec = _mm512_set1_ps(args->x_scale);
    
    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP32 values from x and y.
        __m512 x_vec = _mm512_loadu_ps(&args->x[i]);
        __m512 y_vec = _mm512_loadu_ps(&args->y[i]);
        
        // Compute: result = y + factor * x.
        __m512 result = _mm512_add_ps(y_vec, _mm512_mul_ps(x_vec, factor_vec));
        
        // Store the result back to y.
        _mm512_storeu_ps(&args->y[i], result);
    }
    
    // Scalar fallback for any remaining elements.
    for (long i = limit; i < end; i++) {
        args->y[i] = args->y[i] + args->x_scale * args->x[i];
    }
    
    return NULL;
}

void *thread_func_no_scale_fp32(void *arg) {
    thread_args *args = (thread_args *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 16;  // Process 16 FP32 elements per 512-bit load.
    long limit = start + (len / chunk) * chunk;

    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP32 values from x and y.
        __m512 x_vec = _mm512_loadu_ps(&args->x[i]);
        __m512 y_vec = _mm512_loadu_ps(&args->y[i]);

        // Perform vector addition.
        __m512 sum_vec = _mm512_add_ps(x_vec, y_vec);

        // Store the result back to y.
        _mm512_storeu_ps(&args->y[i], sum_vec);
    }

    // Fallback: Process any remaining elements with scalar code.
    for (long i = limit; i < end; i++) {
        args->y[i] = args->x[i] + args->y[i];
    }

    return NULL;
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <vector_size> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    long n = atol(argv[1]);
    int num_threads = atoi(argv[2]);
    if (n <= 0 || num_threads <= 0) {
        fprintf(stderr, "Error: vector_size and num_threads must be positive numbers.\n");
        return EXIT_FAILURE;
    }
    
    // Allocate 32-byte aligned memory for fp16 vectors (each element is 2 bytes).
    float *x;
    float *y;
    if (posix_memalign((void **)&x, 32, n * sizeof(float)) != 0 ||
        posix_memalign((void **)&y, 32, n * sizeof(float)) != 0) {
        fprintf(stderr, "Error allocating aligned memory.\n");
        return EXIT_FAILURE;
    }
    
    // Initialize vectors.
    for (long i = 0; i < n; i++) {
        x[i] = 1.0;
        y[i] = 2.0;
    }
    
    // Create an array of thread IDs and thread argument structures.
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_args *targs = malloc(num_threads * sizeof(thread_args));
    if (!threads || !targs) {
        fprintf(stderr, "Error allocating thread structures.\n");
        free(x);
        free(y);
        return EXIT_FAILURE;
    }
    
    // Divide the work evenly among threads.
    long chunk_size = n / num_threads;
    long remainder = n % num_threads;
    long start = 0;

    float x_scale = 1.0f;
    
    for (int i = 0; i < num_threads; i++) {
        targs[i].start = start;
        // Distribute the remainder among the first few threads.
        targs[i].end = start + chunk_size + (i < remainder ? 1 : 0);
        targs[i].x = x;
        targs[i].y = y;
	targs[i].x_scale = x_scale;
        start = targs[i].end;
}	
        
	// Start the timer.
    double start_ms = get_time_in_ms();

for (int i = 0; i < num_threads; i++) {
	if (pthread_create(&threads[i], NULL, thread_func_fp32, &targs[i]) != 0) {
            fprintf(stderr, "Error creating thread %d.\n", i);
            free(x);
            free(y);
            free(threads);
            free(targs);
            return EXIT_FAILURE;
        }
    }
    
    // Wait for all threads to complete.
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    double end_ms = get_time_in_ms();
    double elapsed_ms = end_ms - start_ms;
    
    // Calculate memory traffic: each fp16 element is 2 bytes.
    // For each element, we read from x, read from y, and write to y (total of 3 * 2 bytes).
    double total_bytes = 3.0 * n * sizeof(float);
    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = (total_bytes / elapsed_sec) / 1e9;
    
    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Estimated memory bandwidth: %.3f GB/sec\n", bandwidth_GBps);
    
    // Optionally, verify a few results.
    // Since 1.0 + 2.0 = 3.0, the fp16 bit pattern for 3.0 is 0x4200.
    for (int i = 0; i < 5 && i < n; i++) {
        printf("y[%ld] = %.4f\n", (long)i, y[i]);
    }
    
    free(x);
    free(y);
    free(threads);
    free(targs);
    return EXIT_SUCCESS;
}

