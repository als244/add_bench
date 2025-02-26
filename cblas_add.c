#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
//#include <openblas_config.h>  // for openblas_set_num_threads()

// Returns time in milliseconds
double get_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
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

    // Set the number of threads for OpenBLAS
    openblas_set_num_threads(num_threads);

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

    // Get start time
    double start_ms = get_time_in_ms();

    // Perform vector addition: y = 1.0 * x + y
    // (This is equivalent to adding vector x to y.)
    cblas_saxpy(n, 1.0f, x, 1, y, 1);

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
