#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <pthread.h>

// Data type definitions.
typedef uint16_t fp16;
typedef uint16_t bf16;


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

//------------------------------------------------------------------------------
// FP16 --> FP32 Conversion (Multi-threaded)
//------------------------------------------------------------------------------

// Per-thread argument structure.
typedef struct {
    const fp16 *src;  // Input FP16 array.
    float *dst;       // Output FP32 array.
    size_t start;     // Start index (inclusive).
    size_t end;       // End index (exclusive).
} conv_fp16_to_fp32_args;

// Thread worker: converts a slice [start, end) from FP16 to FP32.
static void *thread_conv_fp16_to_fp32(void *arg) {
    conv_fp16_to_fp32_args *targ = (conv_fp16_to_fp32_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;
    const size_t vec_size = 16;  // Process 16 elements at a time.
    
    // Vectorized loop.
    for (i = start; i + vec_size <= end; i += vec_size) {
        __m256i half_vec = _mm256_loadu_si256((const __m256i*)(targ->src + i));
        __m512 fp32_vec = _mm512_cvtph_ps(half_vec);
        _mm512_storeu_ps(targ->dst + i, fp32_vec);
    }
    // Scalar fallback.
    for (; i < end; i++) {
        targ->dst[i] = half_to_float(targ->src[i]);
    }
    return NULL;
}

// Multi-threaded wrapper for FP16 to FP32 conversion.
void convert_fp16_to_fp32_mt(const fp16 *src, float *dst, size_t n, int num_threads) {
    // If threading overhead is not worthwhile, process in a single thread.
    if (num_threads <= 1 || n < (size_t)num_threads * vec_size) {
        conv_fp16_to_fp32_args args = { src, dst, 0, n };
        thread_conv_fp16_to_fp32(&args);
        return;
    }
    
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    conv_fp16_to_fp32_args *targs = malloc(num_threads * sizeof(conv_fp16_to_fp32_args));
    if (!threads || !targs) { /* handle allocation error */ }
    
    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    
    for (int t = 0; t < num_threads; t++) {
        targs[t].src = src;
        targs[t].dst = dst;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_conv_fp16_to_fp32, &targs[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(targs);
}

//------------------------------------------------------------------------------
// FP32 --> FP16 Conversion (Multi-threaded)
//------------------------------------------------------------------------------

typedef struct {
    const float *src;  // Input FP32 array.
    fp16 *dst;         // Output FP16 array.
    size_t start;
    size_t end;
} conv_fp32_to_fp16_args;

static void *thread_conv_fp32_to_fp16(void *arg) {
    conv_fp32_to_fp16_args *targ = (conv_fp32_to_fp16_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;
    const size_t vec_size = 16;

    for (i = start; i + vec_size <= end; i += vec_size) {
        __m512 fp32_vec = _mm512_loadu_ps(targ->src + i);
        __m256i half_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(targ->dst + i), half_vec);
    }
    for (; i < end; i++) {
        targ->dst[i] = float_to_half(targ->src[i]);
    }
    return NULL;
}

void convert_fp32_to_fp16_mt(const float *src, fp16 *dst, size_t n, int num_threads) {
    if (num_threads <= 1 || n < (size_t)num_threads * vec_size) {
        conv_fp32_to_fp16_args args = { src, dst, 0, n };
        thread_conv_fp32_to_fp16(&args);
        return;
    }

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    conv_fp32_to_fp16_args *targs = malloc(num_threads * sizeof(conv_fp32_to_fp16_args));
    if (!threads || !targs) { /* handle error */ }

    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    for (int t = 0; t < num_threads; t++) {
        targs[t].src = src;
        targs[t].dst = dst;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_conv_fp32_to_fp16, &targs[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(targs);
}


// BF16 load/store wrappers (if not provided by your toolchain).
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

//----------------------------------------------------------
// FP16 --> BF16 Conversion (Multi-threaded)
//----------------------------------------------------------

typedef struct {
    const fp16 *src;
    bf16 *dst;
    size_t start;
    size_t end;
} conv_fp16_to_bf16_args;

static void *thread_conv_fp16_to_bf16(void *arg) {
    conv_fp16_to_bf16_args *targ = (conv_fp16_to_bf16_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;
    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        // Load 16 FP16 values.
        __m256i half_vec = _mm256_loadu_si256((const __m256i*)(targ->src + i));
        // Convert FP16 to FP32.
        __m512 fp32_vec = _mm512_cvtph_ps(half_vec);
        // Convert FP32 to BF16.
        __m256bh bf16_vec = _mm512_cvtneps_pbh(fp32_vec);
        // Store 16 BF16 values.
        _mm256_storeu_bf16(targ->dst + i, bf16_vec);
    }
    for (; i < end; i++) {
        float f = half_to_float(targ->src[i]);
        targ->dst[i] = float_to_bf16(f);
    }
    return NULL;
}

void convert_fp16_to_bf16_mt(const fp16 *src, bf16 *dst, size_t n, int num_threads) {
    if (num_threads <= 1) {
        conv_fp16_to_bf16_args args = { src, dst, 0, n };
        thread_conv_fp16_to_bf16(&args);
        return;
    }
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    conv_fp16_to_bf16_args *targs = malloc(num_threads * sizeof(conv_fp16_to_bf16_args));
    if (!threads || !targs) { /* handle allocation error */ }
    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    for (int t = 0; t < num_threads; t++) {
        targs[t].src = src;
        targs[t].dst = dst;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_conv_fp16_to_bf16, &targs[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(targs);
}

//----------------------------------------------------------
// BF16 --> FP16 Conversion (Multi-threaded)
//----------------------------------------------------------

typedef struct {
    const bf16 *src;
    fp16 *dst;
    size_t start;
    size_t end;
} conv_bf16_to_fp16_args;

static void *thread_conv_bf16_to_fp16(void *arg) {
    conv_bf16_to_fp16_args *targ = (conv_bf16_to_fp16_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;
    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        // Load 16 BF16 values.
        __m256bh bf16_vec = _mm256_loadu_bf16(targ->src + i);
        // Convert BF16 to FP32.
        __m512 fp32_vec = _mm512_cvtpbh_ps(bf16_vec);
        // Convert FP32 to FP16.
        __m256i half_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(targ->dst + i), half_vec);
    }
    for (; i < end; i++) {
        float f = bf16_to_float(targ->src[i]);
        targ->dst[i] = float_to_half(f);
    }
    return NULL;
}

void convert_bf16_to_fp16_mt(const bf16 *src, fp16 *dst, size_t n, int num_threads) {
    if (num_threads <= 1) {
        conv_bf16_to_fp16_args args = { src, dst, 0, n };
        thread_conv_bf16_to_fp16(&args);
        return;
    }
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    conv_bf16_to_fp16_args *targs = malloc(num_threads * sizeof(conv_bf16_to_fp16_args));
    if (!threads || !targs) { /* handle allocation error */ }
    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    for (int t = 0; t < num_threads; t++) {
        targs[t].src = src;
        targs[t].dst = dst;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_conv_bf16_to_fp16, &targs[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(targs);
}

//----------------------------------------------------------
// FP32 --> BF16 Conversion (Multi-threaded)
//----------------------------------------------------------

typedef struct {
    const float *src;
    bf16 *dst;
    size_t start;
    size_t end;
} conv_fp32_to_bf16_args;

static void *thread_conv_fp32_to_bf16(void *arg) {
    conv_fp32_to_bf16_args *targ = (conv_fp32_to_bf16_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;
    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        __m512 fp32_vec = _mm512_loadu_ps(targ->src + i);
        __m256bh bf16_vec = _mm512_cvtneps_pbh(fp32_vec);
        _mm256_storeu_bf16(targ->dst + i, bf16_vec);
    }
    for (; i < end; i++) {
        targ->dst[i] = float_to_bf16(targ->src[i]);
    }
    return NULL;
}

void convert_fp32_to_bf16_mt(const float *src, bf16 *dst, size_t n, int num_threads) {
    if (num_threads <= 1) {
        conv_fp32_to_bf16_args args = { src, dst, 0, n };
        thread_conv_fp32_to_bf16(&args);
        return;
    }
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    conv_fp32_to_bf16_args *targs = malloc(num_threads * sizeof(conv_fp32_to_bf16_args));
    if (!threads || !targs) { /* handle allocation error */ }
    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    for (int t = 0; t < num_threads; t++) {
        targs[t].src = src;
        targs[t].dst = dst;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_conv_fp32_to_bf16, &targs[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(targs);
}

//----------------------------------------------------------
// BF16 --> FP32 Conversion (Multi-threaded)
//----------------------------------------------------------

typedef struct {
    const bf16 *src;
    float *dst;
    size_t start;
    size_t end;
} conv_bf16_to_fp32_args;

static void *thread_conv_bf16_to_fp32(void *arg) {
    conv_bf16_to_fp32_args *targ = (conv_bf16_to_fp32_args *)arg;
    size_t start = targ->start, end = targ->end;
    size_t i;
    const size_t vec_size = 16;
    for (i = start; i + vec_size <= end; i += vec_size) {
        __m256bh bf16_vec = _mm256_loadu_bf16(targ->src + i);
        __m512 fp32_vec = _mm512_cvtpbh_ps(bf16_vec);
        _mm512_storeu_ps(targ->dst + i, fp32_vec);
    }
    for (; i < end; i++) {
        targ->dst[i] = bf16_to_float(targ->src[i]);
    }
    return NULL;
}

void convert_bf16_to_fp32_mt(const bf16 *src, float *dst, size_t n, int num_threads) {
    if (num_threads <= 1) {
        conv_bf16_to_fp32_args args = { src, dst, 0, n };
        thread_conv_bf16_to_fp32(&args);
        return;
    }
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    conv_bf16_to_fp32_args *targs = malloc(num_threads * sizeof(conv_bf16_to_fp32_args));
    if (!threads || !targs) { /* handle allocation error */ }
    size_t base_chunk = n / num_threads;
    size_t rem = n % num_threads;
    size_t start = 0;
    for (int t = 0; t < num_threads; t++) {
        targs[t].src = src;
        targs[t].dst = dst;
        targs[t].start = start;
        targs[t].end = start + base_chunk + (t < rem ? 1 : 0);
        start = targs[t].end;
        pthread_create(&threads[t], NULL, thread_conv_bf16_to_fp32, &targs[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(targs);
}



