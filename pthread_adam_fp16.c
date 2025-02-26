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

/*
 * Helper functions to convert between fp16 (stored as uint16_t) and float.
 * The half-precision format is assumed to follow IEEE 754.
 */

static inline float bf16_to_float(uint16_t a) {
    // bf16 is stored in the upper 16 bits of a float.
    uint32_t bits = ((uint32_t)a) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    // Round-to-nearest: add 0x8000 (1 << 15) before truncating.
    bits += 0x8000;
    uint16_t b = (uint16_t)(bits >> 16);
    return b;
}

static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;

    if(exp == 0) {
        if(mant == 0) {
            // Zero.
            f = sign;
        } else {
            // Subnormal number; normalize it.
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;               // Adjust exponent (it was decremented one time too many)
            mant &= ~0x0400;     // Clear the leading 1 that was shifted out
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if(exp == 31) {
        // Inf or NaN.
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        // Normalized number.
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float ret;
    memcpy(&ret, &f, sizeof(ret));
    return ret;
}

static inline uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));

    uint16_t sign = (x >> 16) & 0x8000;
    int32_t exp   = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    uint16_t h;

    if(exp <= 0) {
        // Handle subnormals and zeros.
        if(exp < -10) {
            // Too small becomes zero.
            h = sign;
        } else {
            // Subnormal: add the implicit 1 and shift right.
            mant = (mant | 0x00800000) >> (1 - exp);
            // Rounding: add 0x00001000 (if bit 12 is set) for round-to-nearest.
            if(mant & 0x00001000)
                mant += 0x00002000;
            h = sign | (mant >> 13);
        }
    } else if(exp >= 31) {
        // Overflow: return infinity.
        h = sign | 0x7C00;
    } else {
        // Normalized number.
        // Rounding.
        if(mant & 0x00001000)
            mant += 0x00002000;
        h = sign | (exp << 10) | (mant >> 13);
    }

    return h;
}

#ifdef __AVX512FP16__
// Adam thread arguments.
typedef struct {
    uint16_t *w;       // Parameter array (FP16)
    uint16_t *m;       // First moment array (FP16)
    uint16_t *v;       // Second moment array (FP16)
    uint16_t *g;       // Gradient array (FP16)
    long start;
    long end;
    float beta1;
    float beta2;
    float lr;
    float epsilon;
} adam_thread_args;

void *adam_fp16_avx512(void *arg) {
    adam_thread_args *args = (adam_thread_args *)arg;
    long start = args->start;
    long end = args->end;
    long len = end - start;
    // Process 16 FP16 elements at a time.
    long chunk = 16;
    long limit = start + (len / chunk) * chunk;
    
    // Precompute factors.
    float one_minus_beta1 = 1.0f - args->beta1;
    float one_minus_beta2 = 1.0f - args->beta2;
    
    // Broadcast scalars into 512-bit vectors.
    __m512 beta1_vec = _mm512_set1_ps(args->beta1);
    __m512 beta2_vec = _mm512_set1_ps(args->beta2);
    __m512 one_minus_beta1_vec = _mm512_set1_ps(one_minus_beta1);
    __m512 one_minus_beta2_vec = _mm512_set1_ps(one_minus_beta2);
    __m512 lr_vec = _mm512_set1_ps(args->lr);
    __m512 epsilon_vec = _mm512_set1_ps(args->epsilon);
    
    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP16 elements from each array as 256-bit integers.
        __m256i w_half = _mm256_loadu_si256((__m256i*)(&args->w[i]));
        __m256i m_half = _mm256_loadu_si256((__m256i*)(&args->m[i]));
        __m256i v_half = _mm256_loadu_si256((__m256i*)(&args->v[i]));
        __m256i g_half = _mm256_loadu_si256((__m256i*)(&args->g[i]));
        
        // Convert from FP16 to FP32.
        __m512 w_f = _mm512_cvtph_ps(w_half);
        __m512 m_f = _mm512_cvtph_ps(m_half);
        __m512 v_f = _mm512_cvtph_ps(v_half);
        __m512 g_f = _mm512_cvtph_ps(g_half);
        
        // Adam update:
        // m = beta1 * m + (1 - beta1) * g
        m_f = _mm512_add_ps(_mm512_mul_ps(beta1_vec, m_f),
                            _mm512_mul_ps(one_minus_beta1_vec, g_f));
        
        // v = beta2 * v + (1 - beta2) * (g * g)
        __m512 g2_f = _mm512_mul_ps(g_f, g_f);
        v_f = _mm512_add_ps(_mm512_mul_ps(beta2_vec, v_f),
                            _mm512_mul_ps(one_minus_beta2_vec, g2_f));
        
        // Compute update: update = lr * m / (sqrt(v) + epsilon)
        __m512 sqrt_v = _mm512_sqrt_ps(v_f);
        __m512 denom = _mm512_add_ps(sqrt_v, epsilon_vec);
        __m512 update = _mm512_div_ps(m_f, denom);
        update = _mm512_mul_ps(lr_vec, update);
        
        // Update parameters: w = w - update
        w_f = _mm512_sub_ps(w_f, update);
        
        // Convert FP32 values back to FP16.
        __m256i w_half_new = _mm512_cvtps_ph(w_f, _MM_FROUND_TO_NEAREST_INT);
        __m256i m_half_new = _mm512_cvtps_ph(m_f, _MM_FROUND_TO_NEAREST_INT);
        __m256i v_half_new = _mm512_cvtps_ph(v_f, _MM_FROUND_TO_NEAREST_INT);
        
        // Store the updated FP16 values back to memory.
        _mm256_storeu_si256((__m256i*)(&args->w[i]), w_half_new);
        _mm256_storeu_si256((__m256i*)(&args->m[i]), m_half_new);
        _mm256_storeu_si256((__m256i*)(&args->v[i]), v_half_new);
        // Note: Gradients (args->g) typically are not updated.
    }
    
    // Process any remaining elements using scalar code.
    for (long i = limit; i < end; i++) {
        // Convert FP16 to FP32 (using your provided conversion functions).
        float w = fp16_to_float(args->w[i]);
        float m = fp16_to_float(args->m[i]);
        float v = fp16_to_float(args->v[i]);
        float g = fp16_to_float(args->g[i]);
        
        m = args->beta1 * m + one_minus_beta1 * g;
        v = args->beta2 * v + one_minus_beta2 * (g * g);
        float update = args->lr * m / (sqrtf(v) + args->epsilon);
        w = w - update;
        
        // Convert FP32 back to FP16.
        args->w[i] = float_to_fp16(w);
        args->m[i] = float_to_fp16(m);
        args->v[i] = float_to_fp16(v);
    }
    
    return NULL;
}
#endif


#ifdef __AVX512BF16__

#ifndef _mm256_loadu_bf16
static inline __m256bh my_mm256_loadu_bf16(const void *addr) {
    return (__m256bh)_mm256_loadu_si256((const __m256i *)addr);
}
#define _mm256_loadu_bf16(addr) my_mm256_loadu_bf16(addr)
#endif

#ifndef _mm256_storeu_bf16
static inline void my_mm256_storeu_bf16(void *addr, __m256bh a) {
    _mm256_storeu_si256((__m256i *)addr, (__m256i)a);
}
#define _mm256_storeu_bf16(addr, a) my_mm256_storeu_bf16(addr, a)
#endif

void *adam_bf16_avx512(void *arg) {
    adam_thread_args *args = (adam_thread_args *)arg;
    long start = args->start;
    long end = args->end;
    long len = end - start;
    // Process 16 BF16 elements per iteration.
    long chunk = 16;
    long limit = start + (len / chunk) * chunk;
    
    float one_minus_beta1 = 1.0f - args->beta1;
    float one_minus_beta2 = 1.0f - args->beta2;
    
    // Broadcast scalar values into 512-bit vectors.
    __m512 beta1_vec          = _mm512_set1_ps(args->beta1);
    __m512 beta2_vec          = _mm512_set1_ps(args->beta2);
    __m512 one_minus_beta1_vec= _mm512_set1_ps(one_minus_beta1);
    __m512 one_minus_beta2_vec= _mm512_set1_ps(one_minus_beta2);
    __m512 lr_vec             = _mm512_set1_ps(args->lr);
    __m512 epsilon_vec        = _mm512_set1_ps(args->epsilon);
    
    for (long i = start; i < limit; i += chunk) {
        // Load 16 BF16 elements from each array using BF16-specific load.
        __m256bh w_bf16 = _mm256_loadu_bf16(&args->w[i]);
        __m256bh m_bf16 = _mm256_loadu_bf16(&args->m[i]);
        __m256bh v_bf16 = _mm256_loadu_bf16(&args->v[i]);
        __m256bh g_bf16 = _mm256_loadu_bf16(&args->g[i]);
        
        // Convert BF16 data to single-precision FP32.
        __m512 w_f = _mm512_cvtpbh_ps(w_bf16);
        __m512 m_f = _mm512_cvtpbh_ps(m_bf16);
        __m512 v_f = _mm512_cvtpbh_ps(v_bf16);
        __m512 g_f = _mm512_cvtpbh_ps(g_bf16);
        
        // Adam update:
        // m = beta1 * m + (1 - beta1) * g
        m_f = _mm512_add_ps(_mm512_mul_ps(beta1_vec, m_f),
                            _mm512_mul_ps(one_minus_beta1_vec, g_f));
        
        // v = beta2 * v + (1 - beta2) * (g * g)
        __m512 g2_f = _mm512_mul_ps(g_f, g_f);
        v_f = _mm512_add_ps(_mm512_mul_ps(beta2_vec, v_f),
                            _mm512_mul_ps(one_minus_beta2_vec, g2_f));
        
        // Compute update = lr * m / (sqrt(v) + epsilon)
        __m512 sqrt_v = _mm512_sqrt_ps(v_f);
        __m512 denom = _mm512_add_ps(sqrt_v, epsilon_vec);
        __m512 update = _mm512_div_ps(m_f, denom);
        update = _mm512_mul_ps(lr_vec, update);
        
        // Update parameters: w = w - update
        w_f = _mm512_sub_ps(w_f, update);
        
        // Convert FP32 results back to BF16.
        __m256bh w_bf16_new = _mm512_cvtneps_pbh(w_f);
        __m256bh m_bf16_new = _mm512_cvtneps_pbh(m_f);
        __m256bh v_bf16_new = _mm512_cvtneps_pbh(v_f);
        
        // Store the updated BF16 data.
        _mm256_storeu_bf16(&args->w[i], w_bf16_new);
        _mm256_storeu_bf16(&args->m[i], m_bf16_new);
        _mm256_storeu_bf16(&args->v[i], v_bf16_new);
        // Note: Gradients (args->g) are typically left unchanged.
    }
    
    // Scalar fallback for remaining elements.
    for (long i = limit; i < end; i++) {
        float w = bf16_to_float(args->w[i]);
        float m = bf16_to_float(args->m[i]);
        float v = bf16_to_float(args->v[i]);
        float g = bf16_to_float(args->g[i]);
        
        m = args->beta1 * m + one_minus_beta1 * g;
        v = args->beta2 * v + one_minus_beta2 * (g * g);
        float update = args->lr * m / (sqrtf(v) + args->epsilon);
        w = w - update;
        
        args->w[i] = float_to_bf16(w);
        args->m[i] = float_to_bf16(m);
        args->v[i] = float_to_bf16(v);
    }
    
    return NULL;
}

#endif

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
    uint16_t *w;
    uint16_t *g;
    uint16_t *m;
    uint16_t *v;
    if (posix_memalign((void **)&w, 32, n * sizeof(uint16_t)) != 0 ||
        posix_memalign((void **)&g, 32, n * sizeof(uint16_t)) != 0 ||
      	posix_memalign((void **)&m, 32, n * sizeof(uint16_t)) != 0 ||
        posix_memalign((void **)&v, 32, n * sizeof(uint16_t)) != 0) {
        fprintf(stderr, "Error allocating aligned memory.\n");
        return EXIT_FAILURE;
    }
    
    // Initialize vectors.
   for (long i = 0; i < n; i++) {
    w[i] = 0x3400;  // 0.25 in FP16
    g[i] = 0x2E66;  // ~0.1 in FP16
    m[i] = 0x0000;  // 0.0 in FP16
    v[i] = 0x0000;  // 0.0 in FP16
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
	if (pthread_create(&threads[i], NULL, adam_fp16_avx512, &targs[i]) != 0) {
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
    double total_bytes = 7.0 * n * sizeof(uint16_t);
    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = (total_bytes / elapsed_sec) / 1e9;
    
    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Estimated memory bandwidth: %.3f GB/sec\n", bandwidth_GBps);
    
    // Optionally, verify a few results.
    // Since 1.0 + 2.0 = 3.0, the fp16 bit pattern for 3.0 is 0x4200.
    for (int i = 0; i < 5 && i < n; i++) {
        printf("w[%ld] = 0x%04X\n", (long)i, w[i]);
    }
    
    free(w);
    free(g);
    free(m);
    free(v);
    free(threads);
    free(targs);
    return EXIT_SUCCESS;
}

