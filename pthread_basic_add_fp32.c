#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// Returns time in milliseconds
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
} thread_args;

// Thread function that performs vector addition on its assigned slice.
void *thread_func(void *arg) {
    thread_args *args = (thread_args *)arg;
    long start = args->start;
    long end = args->end;
    
    float * x = args -> x;
    float * y = args -> y;
        
    // Process any remaining elements with scalar code.
    for (long i = start; i < end; i++) {
        y[i] = x[i] + y[i];
    }
    
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <vector_size> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse arguments
    long n = atol(argv[1]);
    int num_threads = atoi(argv[2]);
    if (n <= 0 || num_threads <= 0) {
        fprintf(stderr, "Error: vector_size and num_threads must be positive numbers.\n");
        return EXIT_FAILURE;
    }


    // Allocate memory for two float vectors
    float *x = (float*)malloc(n * sizeof(float));
    float *y = (float*)malloc(n * sizeof(float));
    if (!x || !y) {
        fprintf(stderr, "Error allocating memory.\n");
        free(x);
        free(y);
        return EXIT_FAILURE;
    }

    // Initialize vectors with dummy data (for example, all ones)
    for (long i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
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

     for (int i = 0; i < num_threads; i++) {
        targs[i].start = start;
        // Distribute the remainder among the first few threads.
        targs[i].end = start + chunk_size + (i < remainder ? 1 : 0);
        targs[i].x = x;
        targs[i].y = y;
        start = targs[i].end;
    }

    // Get start time
    double start_ms = get_time_in_ms();

     for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, thread_func, &targs[i]) != 0) {
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

    // Get end time
    double end_ms = get_time_in_ms();
    double elapsed_ms = end_ms - start_ms;

    // Calculate memory bandwidth.
    // Total data moved is: read x (n floats) + read y (n floats) + write y (n floats)
    // Total bytes = 3 * n * sizeof(float)
    double total_bytes = 3.0 * n * sizeof(float);
    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = (total_bytes / elapsed_sec) / 1e9;

    // Print elapsed time and estimated memory bandwidth
    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Estimated memory bandwidth: %.3f GB/sec\n", bandwidth_GBps);

    // Optionally, verify a few results (for testing)
    // For our initialization, each y[i] should be 1.0 + 2.0 = 3.0.
    for (int i = 0; i < 5 && i < n; i++) {
        printf("y[%d] = %.1f\n", i, y[i]);
    }

    free(x);
    free(y);
    return EXIT_SUCCESS;
}

