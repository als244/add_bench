#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <immintrin.h>

// Timing helper: returns current time in milliseconds.
double get_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}


// Adam thread arguments.
typedef struct {
    float *w;       // Parameter array (FP16)
    float *m;       // First moment array (FP16)
    float *v;       // Second moment array (FP16)
    float *g;       // Gradient array (FP16)
    long start;
    long end;
    float beta1;
    float beta2;
    float lr;
    float epsilon;
} adam_thread_args;


void *adam_fp32_avx512(void *arg) {
    adam_thread_args *args = (adam_thread_args *)arg;
    long start = args->start;
    long end = args->end;
    long len = end - start;
    // Process 16 floats per iteration (512 bits / 32 bits per float).
    long chunk = 16;
    long limit = start + (len / chunk) * chunk;
    
    // Broadcast scalars into 512-bit vectors.
    __m512 beta1_vec           = _mm512_set1_ps(args->beta1);
    __m512 beta2_vec           = _mm512_set1_ps(args->beta2);
    __m512 one_minus_beta1_vec = _mm512_set1_ps(1.0f - args->beta1);
    __m512 one_minus_beta2_vec = _mm512_set1_ps(1.0f - args->beta2);
    __m512 lr_vec              = _mm512_set1_ps(args->lr);
    __m512 epsilon_vec         = _mm512_set1_ps(args->epsilon);
    
    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP32 values from each array.
        __m512 w_vec = _mm512_loadu_ps(&args->w[i]);
        __m512 m_vec = _mm512_loadu_ps(&args->m[i]);
        __m512 v_vec = _mm512_loadu_ps(&args->v[i]);
        __m512 g_vec = _mm512_loadu_ps(&args->g[i]);
        
        // Update first moment: m = beta1 * m + (1-beta1) * g
        m_vec = _mm512_add_ps(_mm512_mul_ps(beta1_vec, m_vec),
                              _mm512_mul_ps(one_minus_beta1_vec, g_vec));
        
        // Update second moment: v = beta2 * v + (1-beta2) * (g*g)
        __m512 g2_vec = _mm512_mul_ps(g_vec, g_vec);
        v_vec = _mm512_add_ps(_mm512_mul_ps(beta2_vec, v_vec),
                              _mm512_mul_ps(one_minus_beta2_vec, g2_vec));
        
        // Compute update: update = lr * m / (sqrt(v) + epsilon)
        __m512 sqrt_v = _mm512_sqrt_ps(v_vec);
        __m512 denom = _mm512_add_ps(sqrt_v, epsilon_vec);
        __m512 update = _mm512_div_ps(m_vec, denom);
        update = _mm512_mul_ps(lr_vec, update);
        
        // Update weights: w = w - update
        w_vec = _mm512_sub_ps(w_vec, update);
        
        // Store results back.
        _mm512_storeu_ps(&args->w[i], w_vec);
        _mm512_storeu_ps(&args->m[i], m_vec);
        _mm512_storeu_ps(&args->v[i], v_vec);
        // Typically, gradients (args->g) remain unchanged.
    }
    
    // Process remaining elements with scalar code.
    for (long i = limit; i < end; i++) {
        float w = args->w[i];
        float m = args->m[i];
        float v = args->v[i];
        float g = args->g[i];
        
        m = args->beta1 * m + (1.0f - args->beta1) * g;
        v = args->beta2 * v + (1.0f - args->beta2) * (g * g);
        float update = args->lr * m / (sqrtf(v) + args->epsilon);
        w = w - update;
        
        args->w[i] = w;
        args->m[i] = m;
        args->v[i] = v;
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
    float *w;
    float *g;
    float *m;
    float *v;
    if (posix_memalign((void **)&w, 32, n * sizeof(float)) != 0 ||
        posix_memalign((void **)&g, 32, n * sizeof(float)) != 0 ||
      	posix_memalign((void **)&m, 32, n * sizeof(float)) != 0 ||
        posix_memalign((void **)&v, 32, n * sizeof(float)) != 0) {
        fprintf(stderr, "Error allocating aligned memory.\n");
        return EXIT_FAILURE;
    }
    
    // Initialize vectors.
   for (long i = 0; i < n; i++) {
    w[i] = 0.25;
    g[i] = 0.1;
    m[i] = 0.0;
    v[i] = 0.0;  
}

    // Create an array of thread IDs and thread argument structures.
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    adam_thread_args *targs = malloc(num_threads * sizeof(adam_thread_args));
    if (!threads || !targs) {
        fprintf(stderr, "Error allocating thread structures.\n");
        free(w);
        free(g);
	free(m);
	free(v);
        return EXIT_FAILURE;
    }
    
	
    float beta1 = 0.9;
    float beta2 = 0.999;
    float lr = 0.001;
    float epsilon = 1e-8;


    // Divide the work evenly among threads.
    long chunk_size = n / num_threads;
    long remainder = n % num_threads;
    long start = 0;
    
    for (int i = 0; i < num_threads; i++) {
        targs[i].start = start;
        // Distribute the remainder among the first few threads.
        targs[i].end = start + chunk_size + (i < remainder ? 1 : 0);
        targs[i].w = w;
        targs[i].g = g;
	targs[i].v = v;
	targs[i].m = m;
	targs[i].beta1 = beta1;
	targs[i].beta2 = beta2;
	targs[i].lr = lr;
	targs[i].epsilon = epsilon;
        start = targs[i].end;
}	
        
	// Start the timer.
    double start_ms = get_time_in_ms();

for (int i = 0; i < num_threads; i++) {
	if (pthread_create(&threads[i], NULL, adam_fp32_avx512, &targs[i]) != 0) {
            fprintf(stderr, "Error creating thread %d.\n", i);
            free(w);
	    free(g);
	    free(m);
            free(v);
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
    // For each element, we read 4 arrays and write back 3
    double total_bytes = 7.0 * n * sizeof(float);
    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = (total_bytes / elapsed_sec) / 1e9;
    
    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Estimated memory bandwidth: %.3f GB/sec\n", bandwidth_GBps);
    
    // Optionally, verify a few results.
    // Since 1.0 + 2.0 = 3.0, the fp16 bit pattern for 3.0 is 0x4200.
    for (int i = 0; i < 5 && i < n; i++) {
        printf("w[%ld] = %.4f\n", (long)i, w[i]);
    }
    
    free(w);
    free(g);
    free(m);
    free(v);
    free(threads);
    free(targs);
    return EXIT_SUCCESS;
}

