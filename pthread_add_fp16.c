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


// Structure to hold thread arguments.
typedef struct {
    long start;      // starting index (inclusive)
    long end;        // ending index (exclusive)
    uint16_t *x;
    uint16_t *y;
    float x_scale;
} thread_args;

#ifdef __AVX512FP16__
void *thread_func_fp16(void *arg) {
    thread_args *args = (thread_args *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 16;  // 16 FP16 elements per 256-bit load
    long limit = start + (len / chunk) * chunk;

    // Broadcast the scalar factor into a 512-bit vector.
    __m512 factor_vec = _mm512_set1_ps(args->x_scale);

    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP16 values from x and y as 256-bit integer vectors.
        __m256i x_half = _mm256_loadu_si256((__m256i*)(&args->x[i]));
        __m256i y_half = _mm256_loadu_si256((__m256i*)(&args->y[i]));

        // Convert the FP16 values to FP32.
        __m512 x_f = _mm512_cvtph_ps(x_half);
        __m512 y_f = _mm512_cvtph_ps(y_half);

        // Multiply x by the scalar factor and add to y.
        __m512 result_f = _mm512_add_ps(y_f, _mm512_mul_ps(x_f, factor_vec));

        // Convert the FP32 result back to FP16.
        __m256i res_half = _mm512_cvtps_ph(result_f, _MM_FROUND_TO_NEAREST_INT);

        // Store the result back to y.
        _mm256_storeu_si256((__m256i*)(&args->y[i]), res_half);
    }

    // Scalar fallback for any remaining elements.
    for (long i = limit; i < end; i++) {
        float xf = fp16_to_float(args->x[i]);
        float yf = fp16_to_float(args->y[i]);
        float result = yf + args->x_scale * xf;
        args->y[i] = float_to_fp16(result);
    }

    return NULL;
}


void *thread_func_no_scale_fp16(void *arg) {
    thread_args *args = (thread_args *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    // Process 16 FP16 (half-precision) elements per iteration.
    long chunk = 16;
    long limit = start + (len / chunk) * chunk;

    for (long i = start; i < limit; i += chunk) {
        // Load 16 FP16 values from x and y as 256-bit integer vectors.
        __m256i x_half = _mm256_loadu_si256((__m256i*)(&args->x[i]));
        __m256i y_half = _mm256_loadu_si256((__m256i*)(&args->y[i]));

        // Convert FP16 to FP32.
        __m512 x_f = _mm512_cvtph_ps(x_half);
        __m512 y_f = _mm512_cvtph_ps(y_half);

        // Perform vector addition.
        __m512 sum_f = _mm512_add_ps(x_f, y_f);

        // Convert the result back to FP16.
        __m256i res_half = _mm512_cvtps_ph(sum_f, _MM_FROUND_TO_NEAREST_INT);

        // Store the result back to y.
        _mm256_storeu_si256((__m256i*)(&args->y[i]), res_half);
    }

    // Fallback: Process any remaining elements with scalar code.
    for (long i = limit; i < end; i++) {
        float xf = fp16_to_float(args->x[i]);
        float yf = fp16_to_float(args->y[i]);
        float sum = xf + yf;
        args->y[i] = float_to_fp16(sum);
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

void *thread_func_bf16(void *arg) {
    thread_args *args = (thread_args *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    long chunk = 16;  // Process 16 BF16 elements per iteration.
    long limit = start + (len / chunk) * chunk;

    // Broadcast the scalar factor into a 512-bit vector.
    __m512 factor_vec = _mm512_set1_ps(args->x_scale);

    for (long i = start; i < limit; i += chunk) {
        // Load 16 BF16 values from x and y using BF16 load intrinsic.
        __m256bh x_bf16 = _mm256_loadu_bf16(&args->x[i]);
        __m256bh y_bf16 = _mm256_loadu_bf16(&args->y[i]);

        // Convert BF16 values to FP32.
        __m512 x_f = _mm512_cvtpbh_ps(x_bf16);
        __m512 y_f = _mm512_cvtpbh_ps(y_bf16);

        // Multiply x by the scalar factor and add to y.
        __m512 result_f = _mm512_add_ps(y_f, _mm512_mul_ps(x_f, factor_vec));

        // Convert the FP32 result back to BF16.
        __m256bh res_bf16 = _mm512_cvtneps_pbh(result_f);

        // Store the result back to y.
        _mm256_storeu_bf16(&args->y[i], res_bf16);
    }

    // Scalar fallback: process any remaining elements.
    for (long i = limit; i < end; i++) {
        float xf = bf16_to_float(args->x[i]);
        float yf = bf16_to_float(args->y[i]);
        float result = yf + args->x_scale * xf;
        args->y[i] = float_to_bf16(result);
    }

    return NULL;
}

void *thread_func_no_scale_bf16(void *arg) {
    thread_args *args = (thread_args *)arg;
    long start = args->start;
    long end   = args->end;
    long len   = end - start;
    // Process 16 BF16 elements per iteration.
    long chunk = 16;
    long limit = start + (len / chunk) * chunk;
    
    for (long i = start; i < limit; i += chunk) {
        // Use our BF16-specific load wrappers.
        __m256bh x_bf16 = _mm256_loadu_bf16(&args->x[i]);
        __m256bh y_bf16 = _mm256_loadu_bf16(&args->y[i]);
        
        // Convert BF16 to FP32.
        __m512 x_f = _mm512_cvtpbh_ps(x_bf16);
        __m512 y_f = _mm512_cvtpbh_ps(y_bf16);
        
        // Perform vector addition.
        __m512 sum_f = _mm512_add_ps(x_f, y_f);
        
        // Convert the result back to BF16.
        __m256bh res_bf16 = _mm512_cvtneps_pbh(sum_f);
        
        // Store the BF16 result using our store wrapper.
        _mm256_storeu_bf16(&args->y[i], res_bf16);
    }
    
    // Scalar fallback for any remaining elements.
    for (long i = limit; i < end; i++) {
        float xf = bf16_to_float(args->x[i]);
        float yf = bf16_to_float(args->y[i]);
        float sum = xf + yf;
        args->y[i] = float_to_bf16(sum);
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
    uint16_t *x;
    uint16_t *y;
    if (posix_memalign((void **)&x, 32, n * sizeof(uint16_t)) != 0 ||
        posix_memalign((void **)&y, 32, n * sizeof(uint16_t)) != 0) {
        fprintf(stderr, "Error allocating aligned memory.\n");
        return EXIT_FAILURE;
    }
    
    // Initialize vectors.
    // For fp16, 1.0 is 0x3C00 and 2.0 is 0x4000.
    for (long i = 0; i < n; i++) {
        x[i] = 0x3C00;  // 1.0
        y[i] = 0x4000;  // 2.0
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
	if (pthread_create(&threads[i], NULL, thread_func_fp16, &targs[i]) != 0) {
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
    double total_bytes = 3.0 * n * sizeof(uint16_t);
    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = (total_bytes / elapsed_sec) / 1e9;
    
    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Estimated memory bandwidth: %.3f GB/sec\n", bandwidth_GBps);
    
    // Optionally, verify a few results.
    // Since 1.0 + 2.0 = 3.0, the fp16 bit pattern for 3.0 is 0x4200.
    for (int i = 0; i < 5 && i < n; i++) {
        printf("y[%ld] = 0x%04X\n", (long)i, y[i]);
    }
    
    free(x);
    free(y);
    free(threads);
    free(targs);
    return EXIT_SUCCESS;
}

