#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// Data type definitions.
typedef uint16_t fp16;
typedef uint16_t bf16;

// Forward declarations for scalar conversion functions.
// (Implement these as appropriate for your system.)
extern float half_to_float(fp16 h);
extern fp16 float_to_half(float f);
extern float bf16_to_float(bf16 a);
extern bf16 float_to_bf16(float f);

// BF16 load/store wrappers, if not provided by your toolchain.
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

//------------------------------------------------------------------------------
// FP16 <--> FP32 Conversions
//------------------------------------------------------------------------------

// Convert an array of FP16 values to FP32.
void convert_fp16_to_fp32(const fp16 *src, float *dst, size_t n) {
    size_t i = 0;
    const size_t chunk = 16;
    for (; i + chunk <= n; i += chunk) {
        // Load 16 FP16 values into a 256-bit vector.
        __m256i half_vec = _mm256_loadu_si256((const __m256i*)(src + i));
        // Convert FP16 to FP32.
        __m512 fp32_vec = _mm512_cvtph_ps(half_vec);
        // Store 16 FP32 values.
        _mm512_storeu_ps(dst + i, fp32_vec);
    }
    // Scalar fallback.
    for (; i < n; i++) {
        dst[i] = half_to_float(src[i]);
    }
}

// Convert an array of FP32 values to FP16.
void convert_fp32_to_fp16(const float *src, fp16 *dst, size_t n) {
    size_t i = 0;
    const size_t chunk = 16;
    for (; i + chunk <= n; i += chunk) {
        __m512 fp32_vec = _mm512_loadu_ps(src + i);
        // Convert FP32 to FP16.
        __m256i half_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), half_vec);
    }
    for (; i < n; i++) {
        dst[i] = float_to_half(src[i]);
    }
}

//------------------------------------------------------------------------------
// BF16 <--> FP32 Conversions
//------------------------------------------------------------------------------

// Convert an array of BF16 values to FP32.
void convert_bf16_to_fp32(const bf16 *src, float *dst, size_t n) {
    size_t i = 0;
    const size_t chunk = 16;
    for (; i + chunk <= n; i += chunk) {
        // Load 16 BF16 values into a BF16 vector.
        __m256bh bf16_vec = _mm256_loadu_bf16(src + i);
        // Convert BF16 to FP32.
        __m512 fp32_vec = _mm512_cvtpbh_ps(bf16_vec);
        _mm512_storeu_ps(dst + i, fp32_vec);
    }
    for (; i < n; i++) {
        dst[i] = bf16_to_float(src[i]);
    }
}

// Convert an array of FP32 values to BF16.
void convert_fp32_to_bf16(const float *src, bf16 *dst, size_t n) {
    size_t i = 0;
    const size_t chunk = 16;
    for (; i + chunk <= n; i += chunk) {
        __m512 fp32_vec = _mm512_loadu_ps(src + i);
        // Convert FP32 to BF16.
        __m256bh bf16_vec = _mm512_cvtneps_pbh(fp32_vec);
        _mm256_storeu_bf16(dst + i, bf16_vec);
    }
    for (; i < n; i++) {
        dst[i] = float_to_bf16(src[i]);
    }
}

//------------------------------------------------------------------------------
// FP16 <--> BF16 Conversions (via FP32 intermediate)
//------------------------------------------------------------------------------

// Convert an array of FP16 values to BF16.
void convert_fp16_to_bf16(const fp16 *src, bf16 *dst, size_t n) {
    size_t i = 0;
    const size_t chunk = 16;
    for (; i + chunk <= n; i += chunk) {
        // Load FP16 values.
        __m256i half_vec = _mm256_loadu_si256((const __m256i*)(src + i));
        // Convert FP16 to FP32.
        __m512 fp32_vec = _mm512_cvtph_ps(half_vec);
        // Convert FP32 to BF16.
        __m256bh bf16_vec = _mm512_cvtneps_pbh(fp32_vec);
        _mm256_storeu_bf16(dst + i, bf16_vec);
    }
    for (; i < n; i++) {
        float f = half_to_float(src[i]);
        dst[i] = float_to_bf16(f);
    }
}

// Convert an array of BF16 values to FP16.
void convert_bf16_to_fp16(const bf16 *src, fp16 *dst, size_t n) {
    size_t i = 0;
    const size_t chunk = 16;
    for (; i + chunk <= n; i += chunk) {
        __m256bh bf16_vec = _mm256_loadu_bf16(src + i);
        // Convert BF16 to FP32.
        __m512 fp32_vec = _mm512_cvtpbh_ps(bf16_vec);
        // Convert FP32 to FP16.
        __m256i half_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), half_vec);
    }
    for (; i < n; i++) {
        float f = bf16_to_float(src[i]);
        dst[i] = float_to_half(f);
    }
}

