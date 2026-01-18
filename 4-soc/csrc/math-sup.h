/*
 * The header file supported the fixed-point math operations and types
 * for the triangle rasterization render in `vga.c`.
 * The claculation definition in this code are adapted from 
 * `b3d/math-toolkit.h` library.
 */

#ifndef S3D_MATH_H
#define S3D_MATH_H

#include <limits.h>
// #include <math.h>
#include <stdint.h>
#include <stdbool.h>

/* Fixed-point math operations */

typedef int32_t s3d_fixed_t;
#define S3D_FP_BITS 16
#define S3D_FP_ONE (1LL << S3D_FP_BITS) /* 1LL avoids UB */
#define S3D_FP_HALF (1LL << (S3D_FP_BITS - 1))
#define S3D_FP_EPSILON 16 /* ~0.000244 in Q15.16, for near-zero comparisons */

/* int64_t intermediates required for precision.
 * Use multiplication instead of left-shift to avoid UB with negative values. */
// #define S3D_FP_MUL(a, b) ((s3d_fixed_t) (((int64_t) (a) * (b)) >> S3D_FP_BITS))
// #define S3D_FP_DIV(a, b) 
//     ((b) == 0 ? 0 : (s3d_fixed_t) (((int64_t) (a) * S3D_FP_ONE) / (b)))

/* Fixed-point absolute value - guards against INT32_MIN overflow */
static inline s3d_fixed_t s3d_fp_abs(s3d_fixed_t x)
{
    if (x == INT32_MIN)
        return INT32_MAX; /* -INT32_MIN overflows; clamp to max */
    return x < 0 ? -x : x;
}

/* Consider using RV32I instructions, which doesn't contain M extension */
static inline s3d_fixed_t S3D_FP_MUL(s3d_fixed_t a, s3d_fixed_t b)
{
    bool negative = (a ^ b) < 0;
    uint32_t abs_a = (uint32_t) s3d_fp_abs(a);
    uint32_t abs_b = (uint32_t) s3d_fp_abs(b);
    if (abs_a < abs_b) {
        uint32_t tmp = abs_a;
        abs_a = abs_b;
        abs_b = tmp;
    }

    uint32_t res_hi = 0;
    uint32_t res_lo = 0;
    uint32_t ma_lo = abs_a;
    uint32_t ma_hi = 0;

    while (abs_b != 0) {
        if (abs_b & 1) {
            /* res += ma */
            uint32_t new_lo = res_lo + ma_lo;
            uint32_t carry = (new_lo < res_lo) ? 1 : 0;
            res_lo = new_lo;
            res_hi += ma_hi + carry;
        }
        ma_hi = (ma_hi << 1) | (ma_lo >> 31);
        ma_lo <<= 1;

        abs_b >>= 1;
    }

    uint32_t result = (res_hi << (32 - S3D_FP_BITS)) | (res_lo >> S3D_FP_BITS);
    return negative ? -(s3d_fixed_t)result : (s3d_fixed_t)result;
}

/* Counting leading zeros of a unsigned 32 bit number */
static inline uint32_t s3d_clz(uint32_t x) 
{
    if (x == 0) 
        return 32;

    uint32_t n = 0;

    /* branchless calculating the leading zeros */
    if ((x & 0xFFFF0000) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000) == 0) { n += 8;  x <<= 8;  }
    if ((x & 0xF0000000) == 0) { n += 4;  x <<= 4;  }
    if ((x & 0xC0000000) == 0) { n += 2;  x <<= 2;  }
    if ((x & 0x80000000) == 0) { n += 1; }
    
    return n;
}

/* Consider using RV32I instructions, which doesn't contain M extension */
static inline s3d_fixed_t S3D_FP_DIV(s3d_fixed_t a, s3d_fixed_t b)
{
    if (b == 0)
        return 0;

    bool negative = (a ^ b) < 0;
    uint32_t dividend = (uint32_t) s3d_fp_abs(a);
    uint32_t divisor = (uint32_t) s3d_fp_abs(b);
    uint32_t quotient = 0;
    uint32_t remainder = 0;

    uint32_t lz = s3d_clz(dividend);
    int sb = (31 - lz) + S3D_FP_BITS; // starting bit index

    for (int i = sb; i >= 0; --i) {
        remainder <<= 1;

        if (i > 16) {
            uint32_t real_bit_idx = i - 16;
            uint32_t bit = (dividend >> real_bit_idx) & 1U;
            remainder = (remainder << 1) | bit;
        }
        
        quotient <<= 1;
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= 1;
        }
    }

    return negative ? -(s3d_fixed_t)quotient : (s3d_fixed_t)quotient;
}

#define S3D_FP_FLOOR(f)                                  \
    ((f) & ~((1 << S3D_FP_BITS) - 1)) /* replaces floorf \
                                       */
#define S3D_FP_ADD(a, b) ((a) + (b))
#define S3D_FP_SUB(a, b) ((a) - (b))

/* Use multiplication (not left-shift) to avoid UB with negative values */
// #define S3D_INT_TO_FP(i) ((s3d_fixed_t) ((int64_t) (i) * S3D_FP_ONE))
/* Use bit shifting since this it  */
#define S3D_INT_TO_FP(i) ((s3d_fixed_t)((uint32_t)(i) << S3D_FP_BITS))
#define S3D_FP_TO_INT(f) ((f) >> S3D_FP_BITS)

/* Define scalar type for fixed-point */
typedef s3d_fixed_t s3d_scalar_t;

/* Fixed-point constants */
#define S3D_FP_PI ((s3d_fixed_t) (S3D_FP_DIV((355LL << S3D_FP_BITS), S3D_INT_TO_FP(113))))
#define S3D_FP_PI_HALF (S3D_FP_PI >> 1)
#define S3D_FP_3PI_HALF (S3D_FP_PI + S3D_FP_PI_HALF)
#define S3D_FP_2PI (S3D_FP_PI << 1)
#define S3D_FP_PI_SQ \
    ((s3d_fixed_t) (S3D_FP_MUL(S3D_FP_PI, S3D_FP_PI)))
#define S3D_DEGEN_THRESHOLD 66
#define S3D_CULL_THRESHOLD 655 /* ~0.009946 in Q15.16, for near-zero comparisons */
#define S3D_NEAR_DISTANCE 6554 /* ~0.10000763 in Q15.16, for near-zero comparisons */
#define S3D_FAR_DISTANCE 6553600 /* ~100 in Q15.16, for near-zero comparisons */
#define S3D_CLIP_BUFFER_SIZE 32  /* Maximum triangles in clipping buffer */ 

/* Depth buffer size */
#define S3D_DEPTH_CLEAR INT32_MAX

/* Bhaskara I kernel for x in [0, π], returns positive sine approximation */
static inline s3d_fixed_t s3d_fp_sin_core(s3d_fixed_t x)
{
    s3d_fixed_t xp = S3D_FP_MUL(x, S3D_FP_PI - x);
    s3d_fixed_t denom = S3D_FP_MUL(S3D_INT_TO_FP(5), S3D_FP_PI_SQ) - S3D_FP_MUL(S3D_INT_TO_FP(4), xp);

    if (denom == 0)
        return 0;

    return S3D_FP_DIV(S3D_FP_MUL(S3D_INT_TO_FP(16), xp), denom);
}

/* Bhaskara I sine: sin(x) ≈ 16x(π-x) / (5π² - 4x(π-x)), ~0.3% max error */
static inline s3d_fixed_t s3d_fp_sin(s3d_fixed_t x)
{
    int sign = 1;
    s3d_fixed_t x64 = x; /* change it back to s3d_fixed_t 
                            since the MUL has handled it*/

    /* Handle negative angles - guard against INT32_MIN overflow */
    if (x64 < 0) {
        if (x64 == INT32_MIN)
            x64 = INT32_MAX; /* Clamp to avoid overflow on negation */
        else
            x64 = -x64;
        sign = -sign;
    }

    /* Fast modulo reduction using int64_t (avoids slow loop for large angles)
     */
    // if (x64 >= S3D_FP_2PI) {
    //     x64 %= (int64_t) S3D_FP_2PI;
    // }

    /* RV32I doesn't support modulo, so replace it with manully modulo */
    if (x64 >= S3D_FP_2PI) {
        // x % y = x - (integer(x / y) * y)
        s3d_fixed_t times = (s3d_fixed_t) S3D_FP_MUL((s3d_fixed_t) x64, 
                                                     S3D_FP_DIV(S3D_FP_ONE, S3D_FP_2PI));
        x64 -= S3D_FP_MUL(times, S3D_FP_2PI);
        while (x64 >= S3D_FP_2PI) 
            x64 -= S3D_FP_2PI;
    }

    /* Map (π, 2π) to (0, π) with sign flip */
    s3d_fixed_t angle = (s3d_fixed_t) x64;
    if (angle > S3D_FP_PI) {
        angle -= S3D_FP_PI;
        sign = -sign;
    }

    return S3D_FP_MUL(S3D_INT_TO_FP(sign), s3d_fp_sin_core(angle));
}

/* Compute sine and cosine together to share reduction work */
static inline void s3d_fp_sincos(s3d_fixed_t x,
                                 s3d_fixed_t *sinp,
                                 s3d_fixed_t *cosp)
{
    int sin_sign = 1;
    s3d_fixed_t x64 = x;

    if (x64 < 0) {
        if (x64 == INT32_MIN)
            x64 = INT32_MAX;
        else
            x64 = -x64;
        sin_sign = -1;
    }

    // if (x64 >= S3D_FP_2PI)
    //     x64 %= (int64_t) S3D_FP_2PI;
    if (x64 >= S3D_FP_2PI) {
        // x % y = x - (integer(x / y) * y)
        s3d_fixed_t times = (s3d_fixed_t) S3D_FP_MUL((s3d_fixed_t) x64, 
                                                     S3D_FP_DIV(S3D_FP_ONE, S3D_FP_2PI));
        x64 -= S3D_FP_MUL(times, S3D_FP_2PI);
        while (x64 >= S3D_FP_2PI) 
            x64 -= S3D_FP_2PI;
    }

    s3d_fixed_t angle = (s3d_fixed_t) x64;

    /* Sine */
    s3d_fixed_t sin_angle = angle;
    if (sin_angle > S3D_FP_PI) {
        sin_angle -= S3D_FP_PI;
        sin_sign = -sin_sign;
    }
    s3d_fixed_t sin_val = s3d_fp_sin_core(sin_angle);
    if (sinp)
        *sinp = (sin_sign == 1) ? sin_val : -sin_val;

    /* Cosine via quadrant mapping to [0, π/2] */
    int quadrant = (int) S3D_FP_DIV(x64, S3D_FP_PI_HALF);
    s3d_fixed_t cos_angle;
    int cos_sign;

    switch (quadrant) {
    case 0:
        cos_angle = S3D_FP_PI_HALF - angle;
        cos_sign = 1;
        break;
    case 1:
        cos_angle = angle - S3D_FP_PI_HALF;
        cos_sign = -1;
        break;
    case 2:
        cos_angle = S3D_FP_3PI_HALF - angle;
        cos_sign = -1;
        break;
    default:
        cos_angle = angle - S3D_FP_3PI_HALF;
        cos_sign = 1;
        break;
    }

    s3d_fixed_t cos_val = s3d_fp_sin_core(cos_angle);
    if (cosp)
        *cosp = (cos_sign == 1) ? cos_val : -cos_val;
}

static inline s3d_fixed_t s3d_fp_cos(s3d_fixed_t x)
{
    /* Consider the Symmetry properties in cosine: cos(-x) = cos(x) */
    s3d_fixed_t x32 = s3d_fp_abs(x);
    /* sin(x + PI/2) = sin(x + PI/2 - 2PI). 
     * Check for the safe threshold of adding PI/2. */
    const int32_t SAFE_LIMIT = INT32_MAX - S3D_FP_PI_HALF;
    if (x32 > SAFE_LIMIT)
        x32 -= S3D_FP_2PI;
    /* Safely add PI/2 to the given angle.
     * This also represents that the big angle issue would be 
     * handle in the `s3d_fp_sin()` function instead of doing it
     * locally. */
    x32 += S3D_FP_PI_HALF;
    // /* Do addition in int64_t to avoid overflow for large angles near INT32_MAX.
    //  * s3d_fp_sin handles modulo reduction internally. */
    // int64_t x64 = (int64_t) x + (int64_t) S3D_FP_PI_HALF;
    // /* Reduce to int32_t range before calling sin */
    // if (x64 > INT32_MAX)
    //     x64 = x64 % (int64_t) S3D_FP_2PI;
    // else if (x64 < INT32_MIN)
    //     x64 = -((-x64) % (int64_t) S3D_FP_2PI);
    return s3d_fp_sin((s3d_fixed_t) x32);
}

/* tangent with fixed-point calculation (might not be precise) */
static inline s3d_fixed_t s3d_fp_tan(s3d_fixed_t x)
{
    s3d_fixed_t c = s3d_fp_cos(x);
    return (c < S3D_FP_EPSILON && c > -S3D_FP_EPSILON) ? 0 : S3D_FP_DIV(s3d_fp_sin(x), c);
}

/* Integer sqrt on Q16.16: computes floor(sqrt(a)) in fixed-point */
static inline s3d_fixed_t s3d_fp_sqrt(s3d_fixed_t a)
{
    if (a <= 0)
        return 0;

    /* Scale by 2^16 so sqrt preserves fixed-point fraction bits */
    uint64_t n = ((uint64_t) (uint32_t) a) << S3D_FP_BITS;
    uint64_t res = 0;
    uint64_t bit = 1ULL << 62; /* Largest power-of-four starting point */

    while (bit > n)
        bit >>= 2;

    while (bit != 0) {
        if (n >= res + bit) {
            n -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }

    return res > INT32_MAX ? INT32_MAX : (s3d_fixed_t) res;
}

static inline s3d_scalar_t s3d_fp_min(s3d_scalar_t a, s3d_scalar_t b)
{
    return a < b ? a : b;
}

static inline s3d_scalar_t s3d_fp_max(s3d_scalar_t a, s3d_scalar_t b)
{
    return a > b ? a : b;
}

/* Matrix operations (we only consider the case with culling technique included) */

/*
 * Internal vector/matrix types
 */
typedef struct {
    s3d_fixed_t x, y, z, w;
} s3d_vec_t;

typedef struct {
    s3d_fixed_t m[4][4];
} s3d_mat_t;

typedef struct {
    s3d_vec_t p[3];
} s3d_triangle_t;

static inline s3d_fixed_t s3d_vec_dot(s3d_vec_t a, s3d_vec_t b)
{
    return (S3D_FP_ADD(S3D_FP_MUL(a.x, b.x), S3D_FP_ADD(S3D_FP_MUL(a.y, b.y), S3D_FP_MUL(a.z, b.z))));
}

static inline s3d_fixed_t s3d_vec_length(s3d_vec_t v)
{
    return s3d_fp_sqrt(s3d_vec_dot(v, v));
}

static inline s3d_vec_t s3d_vec_add(s3d_vec_t a, s3d_vec_t b)
{
    return (s3d_vec_t){
        S3D_FP_ADD(a.x, b.x),
        S3D_FP_ADD(a.y, b.y),
        S3D_FP_ADD(a.z, b.z),
        S3D_FP_ADD(a.w, b.w)
    };
}

static inline s3d_vec_t s3d_vec_sub(s3d_vec_t a, s3d_vec_t b)
{
    return (s3d_vec_t){
        S3D_FP_SUB(a.x, b.x),
        S3D_FP_SUB(a.y, b.y),
        S3D_FP_SUB(a.z, b.z),
        S3D_FP_SUB(a.w, b.w)
    };
}

static inline s3d_vec_t s3d_vec_mul(s3d_vec_t v, s3d_fixed_t s)
{
    return (s3d_vec_t){
        S3D_FP_MUL(v.x, s),
        S3D_FP_MUL(v.y, s),
        S3D_FP_MUL(v.z, s),
        S3D_FP_MUL(v.w, s)
    };
}

static inline s3d_vec_t s3d_vec_div(s3d_vec_t v, s3d_fixed_t s)
{
    if (s3d_fp_abs(s) < S3D_FP_EPSILON)
        return (s3d_vec_t){v.x, v.y, v.z, S3D_FP_ONE};
    return (s3d_vec_t){
        S3D_FP_DIV(v.x, s),
        S3D_FP_DIV(v.y, s),
        S3D_FP_DIV(v.z, s),
        S3D_FP_ONE
    };
}

static inline s3d_vec_t s3d_vec_neg(s3d_vec_t v)
{
    return (s3d_vec_t){
        -(v.x),
        -(v.y),
        -(v.z),
        v.w
    };
}

static inline s3d_vec_t s3d_vec_cross(s3d_vec_t a, s3d_vec_t b)
{
    return (s3d_vec_t){
        S3D_FP_SUB(S3D_FP_MUL(a.y, b.z), S3D_FP_MUL(a.z, b.y)),
        S3D_FP_SUB(S3D_FP_MUL(a.z, b.x), S3D_FP_MUL(a.x, b.z)),
        S3D_FP_SUB(S3D_FP_MUL(a.x, b.y), S3D_FP_MUL(a.y, b.x)),
        S3D_FP_ONE
    };
}

static inline s3d_vec_t s3d_vec_norm(s3d_vec_t v)
{
    s3d_fixed_t len = s3d_vec_length(v);
    if (len == 0)
        return v; /* Avoid division by zero */
    return (s3d_vec_t){
        S3D_FP_DIV(v.x, len),
        S3D_FP_DIV(v.y, len),
        S3D_FP_DIV(v.z, len),
        v.w
    };
}

static inline s3d_vec_t s3d_vec_lerp(s3d_vec_t a, s3d_vec_t b, s3d_fixed_t t)
{
    return (s3d_vec_t){
        S3D_FP_ADD(a.x, S3D_FP_MUL(S3D_FP_SUB(b.x, a.x), t)),
        S3D_FP_ADD(a.y, S3D_FP_MUL(S3D_FP_SUB(b.y, a.y), t)),
        S3D_FP_ADD(a.z, S3D_FP_MUL(S3D_FP_SUB(b.z, a.z), t)),
        S3D_FP_ADD(a.w, S3D_FP_MUL(S3D_FP_SUB(b.w, a.w), t))
    };
}

static inline s3d_vec_t s3d_mat_row3(s3d_mat_t m, int r)
{
    return (s3d_vec_t){
        m.m[r][0],
        m.m[r][1],
        m.m[r][2],
        0
    };
}

static inline s3d_mat_t s3d_mat_from_rows(s3d_vec_t r0,
                                     s3d_vec_t r1,
                                     s3d_vec_t r2,
                                     s3d_vec_t r3)
{
    return (s3d_mat_t){
        .m = {
            {r0.x, r0.y, r0.z, r0.w},
            {r1.x, r1.y, r1.z, r1.w},
            {r2.x, r2.y, r2.z, r2.w},
            {r3.x, r3.y, r3.z, r3.w}
        }
    };
}

static inline s3d_mat_t s3d_mat_qinv(s3d_mat_t m)
{
    s3d_vec_t m0 = s3d_mat_row3(m, 0);
    s3d_vec_t m1 = s3d_mat_row3(m, 1);
    s3d_vec_t m2 = s3d_mat_row3(m, 2);
    s3d_vec_t t = s3d_mat_row3(m, 3);

    return s3d_mat_from_rows(
        (s3d_vec_t){
            m0.x, m1.x, m2.x, 0
        },
        (s3d_vec_t){
            m0.y, m1.y, m2.y, 0
        },
        (s3d_vec_t){
            m0.z, m1.z, m2.z, 0
        },
        (s3d_vec_t){
            -s3d_vec_dot(t, m0),
            -s3d_vec_dot(t, m1),
            -s3d_vec_dot(t, m2),
            S3D_FP_ONE
        }
    );
}

static inline s3d_mat_t s3d_mat_ident()
{
    return (s3d_mat_t){
        .m = {
            {S3D_FP_ONE, 0, 0, 0},
            {0, S3D_FP_ONE, 0, 0},
            {0, 0, S3D_FP_ONE, 0},
            {0, 0, 0, S3D_FP_ONE}
        }
    };
}

/* Matrix rotation (refers to rotate tranformation algorithm) */

static inline s3d_mat_t s3d_mat_rot_x(s3d_fixed_t a)
{
    return (s3d_mat_t){
        {
            {S3D_FP_ONE, 0, 0, 0},
            {0, s3d_fp_cos(a), s3d_fp_sin(a), 0},
            {0, -s3d_fp_sin(a), s3d_fp_cos(a), 0},
            {0, 0, 0, S3D_FP_ONE}
        }
    };
}

static inline s3d_mat_t s3d_mat_rot_y(s3d_fixed_t a)
{
    return (s3d_mat_t){
        {
            {s3d_fp_cos(a), 0, -s3d_fp_sin(a), 0},
            {0, S3D_FP_ONE, 0, 0},
            {s3d_fp_sin(a), 0, s3d_fp_cos(a), 0},
            {0, 0, 0, S3D_FP_ONE}
        }
    };
}

static inline s3d_mat_t s3d_mat_rot_z(s3d_fixed_t a)
{
    return (s3d_mat_t){
        {
            {s3d_fp_cos(a), s3d_fp_sin(a), 0, 0},
            {-s3d_fp_sin(a), s3d_fp_cos(a), 0, 0},
            {0, 0, S3D_FP_ONE, 0},
            {0, 0, 0, S3D_FP_ONE}
        }
    };
}

static inline s3d_mat_t s3d_mat_proj(int32_t fov,
                                  s3d_fixed_t aspect,
                                  s3d_fixed_t near,
                                  s3d_fixed_t far)
{
    s3d_fixed_t fov_rad = S3D_FP_DIV(S3D_FP_MUL(S3D_INT_TO_FP(fov), S3D_FP_PI),
                                     S3D_FP_2PI);
    s3d_fixed_t f = S3D_FP_DIV(S3D_FP_ONE, s3d_fp_tan(S3D_FP_MUL(fov_rad, S3D_FP_HALF)));
    s3d_fixed_t nf = S3D_FP_SUB(far, near);

    return (s3d_mat_t){
        .m = {
            {S3D_FP_MUL(f, aspect), 0, 0, 0},
            {0, f, 0, 0},
            {0, 0, S3D_FP_DIV(far, nf), S3D_FP_ONE},
            {0, 0, S3D_FP_DIV(S3D_FP_MUL(S3D_FP_MUL(-S3D_FP_ONE, far), near), nf), 0}
        }
    };
}

/* Matrix multiplication operations based on those listed in `math-gen.inc` */

/* MatMul without iteration (what `math-gen.inc` generated) */
static inline s3d_mat_t s3d_mat_mul(s3d_mat_t a, s3d_mat_t b)
{
    return (s3d_mat_t) {
        .m = {
            {
                S3D_FP_MUL(a.m[0][0], b.m[0][0]) + S3D_FP_MUL(a.m[0][1], b.m[1][0]) +
                    S3D_FP_MUL(a.m[0][2], b.m[2][0]) + S3D_FP_MUL(a.m[0][3], b.m[3][0]),
                S3D_FP_MUL(a.m[0][0], b.m[0][1]) + S3D_FP_MUL(a.m[0][1], b.m[1][1]) +
                    S3D_FP_MUL(a.m[0][2], b.m[2][1]) + S3D_FP_MUL(a.m[0][3], b.m[3][1]),
                S3D_FP_MUL(a.m[0][0], b.m[0][2]) + S3D_FP_MUL(a.m[0][1], b.m[1][2]) +
                    S3D_FP_MUL(a.m[0][2], b.m[2][2]) + S3D_FP_MUL(a.m[0][3], b.m[3][2]),
                S3D_FP_MUL(a.m[0][0], b.m[0][3]) + S3D_FP_MUL(a.m[0][1], b.m[1][3]) +
                    S3D_FP_MUL(a.m[0][2], b.m[2][3]) + S3D_FP_MUL(a.m[0][3], b.m[3][3])
            },
            {
                S3D_FP_MUL(a.m[1][0], b.m[0][0]) + S3D_FP_MUL(a.m[1][1], b.m[1][0]) +
                    S3D_FP_MUL(a.m[1][2], b.m[2][0]) + S3D_FP_MUL(a.m[1][3], b.m[3][0]),
                S3D_FP_MUL(a.m[1][0], b.m[0][1]) + S3D_FP_MUL(a.m[1][1], b.m[1][1]) +
                    S3D_FP_MUL(a.m[1][2], b.m[2][1]) + S3D_FP_MUL(a.m[1][3], b.m[3][1]),
                S3D_FP_MUL(a.m[1][0], b.m[0][2]) + S3D_FP_MUL(a.m[1][1], b.m[1][2]) +
                    S3D_FP_MUL(a.m[1][2], b.m[2][2]) + S3D_FP_MUL(a.m[1][3], b.m[3][2]),
                S3D_FP_MUL(a.m[1][0], b.m[0][3]) + S3D_FP_MUL(a.m[1][1], b.m[1][3]) +
                    S3D_FP_MUL(a.m[1][2], b.m[2][3]) + S3D_FP_MUL(a.m[1][3], b.m[3][3])
            },
            {
                S3D_FP_MUL(a.m[2][0], b.m[0][0]) + S3D_FP_MUL(a.m[2][1], b.m[1][0]) +
                    S3D_FP_MUL(a.m[2][2], b.m[2][0]) + S3D_FP_MUL(a.m[2][3], b.m[3][0]),
                S3D_FP_MUL(a.m[2][0], b.m[0][1]) + S3D_FP_MUL(a.m[2][1], b.m[1][1]) +
                    S3D_FP_MUL(a.m[2][2], b.m[2][1]) + S3D_FP_MUL(a.m[2][3], b.m[3][1]),
                S3D_FP_MUL(a.m[2][0], b.m[0][2]) + S3D_FP_MUL(a.m[2][1], b.m[1][2]) +
                    S3D_FP_MUL(a.m[2][2], b.m[2][2]) + S3D_FP_MUL(a.m[2][3], b.m[3][2]), 
                S3D_FP_MUL(a.m[2][0], b.m[0][3]) + S3D_FP_MUL(a.m[2][1], b.m[1][3]) +
                    S3D_FP_MUL(a.m[2][2], b.m[2][3]) + S3D_FP_MUL(a.m[2][3], b.m[3][3])
            },
            {
                S3D_FP_MUL(a.m[3][0], b.m[0][0]) + S3D_FP_MUL(a.m[3][1], b.m[1][0]) +
                    S3D_FP_MUL(a.m[3][2], b.m[2][0]) + S3D_FP_MUL(a.m[3][3], b.m[3][0]),
                S3D_FP_MUL(a.m[3][0], b.m[0][1]) + S3D_FP_MUL(a.m[3][1], b.m[1][1]) +
                    S3D_FP_MUL(a.m[3][2], b.m[2][1]) + S3D_FP_MUL(a.m[3][3], b.m[3][1]),
                S3D_FP_MUL(a.m[3][0], b.m[0][2]) + S3D_FP_MUL(a.m[3][1], b.m[1][2]) +
                    S3D_FP_MUL(a.m[3][2], b.m[2][2]) + S3D_FP_MUL(a.m[3][3], b.m[3][2]),
                S3D_FP_MUL(a.m[3][0], b.m[0][3]) + S3D_FP_MUL(a.m[3][1], b.m[1][3]) +
                    S3D_FP_MUL(a.m[3][2], b.m[2][3]) + S3D_FP_MUL(a.m[3][3], b.m[3][3])
            }
        }
    };
}

static inline s3d_vec_t s3d_mat_mul_vec(s3d_mat_t m, s3d_vec_t v) 
{
    return (s3d_vec_t){
        (S3D_FP_MUL(m.m[0][0], v.x) + S3D_FP_MUL(m.m[0][1], v.y) + S3D_FP_MUL(m.m[0][2], v.z) + S3D_FP_MUL(m.m[0][3], v.w)),
        (S3D_FP_MUL(m.m[1][0], v.x) + S3D_FP_MUL(m.m[1][1], v.y) + S3D_FP_MUL(m.m[1][2], v.z) + S3D_FP_MUL(m.m[1][3], v.w)),
        (S3D_FP_MUL(m.m[2][0], v.x) + S3D_FP_MUL(m.m[2][1], v.y) + S3D_FP_MUL(m.m[2][2], v.z) + S3D_FP_MUL(m.m[2][3], v.w)),
        (S3D_FP_MUL(m.m[3][0], v.x) + S3D_FP_MUL(m.m[3][1], v.y) + S3D_FP_MUL(m.m[3][2], v.z) + S3D_FP_MUL(m.m[3][3], v.w))
    };
}

static inline s3d_mat_t s3d_mat_point_at(s3d_vec_t pos, s3d_vec_t target, s3d_vec_t up)
{
    s3d_vec_t fwd = s3d_vec_sub(target, pos);
    s3d_fixed_t f3d_len_sq = s3d_vec_dot(fwd, fwd);
    if (f3d_len_sq < S3D_FP_EPSILON) {
        return s3d_mat_ident();
    } else {
        s3d_vec_t fwd_n = s3d_vec_norm(fwd);
        s3d_vec_t up_proj = s3d_vec_sub(up, s3d_vec_mul(fwd_n, s3d_vec_dot(up, fwd_n)));
        s3d_fixed_t up_len_sq = s3d_vec_dot(up_proj, up_proj);
        if (up_len_sq < S3D_FP_EPSILON) {
            return s3d_mat_ident();
        } else {
            s3d_vec_t up_n = s3d_vec_norm(up_proj);
            s3d_vec_t right = s3d_vec_cross(up_n, fwd_n);
            return s3d_mat_from_rows(
                (s3d_vec_t){
                    right.x, right.y, right.z, 0
                },
                (s3d_vec_t){
                    up_n.x, up_n.y, up_n.z, 0
                },
                (s3d_vec_t){
                    fwd_n.x, fwd_n.y, fwd_n.z, 0
                },
                (s3d_vec_t){
                    pos.x, pos.y, pos.z, S3D_FP_ONE
                }
            );
        }
    }
}

static inline s3d_vec_t s3d_intersect_plane(s3d_vec_t n,
                                         s3d_fixed_t d,
                                         s3d_vec_t start,
                                         s3d_vec_t end)
{
    s3d_fixed_t ad = s3d_vec_dot(start, n);
    s3d_fixed_t bd = s3d_vec_dot(end, n);
    s3d_fixed_t denom = S3D_FP_SUB(bd, ad);
    if (s3d_fp_abs(denom) < S3D_FP_EPSILON) {
        return start; /* Line is parallel to plane */
    } else {
        s3d_fixed_t t = s3d_fp_min(s3d_fp_max(S3D_FP_DIV(((d) - (ad)), denom), 0), S3D_FP_ONE);
        return s3d_vec_lerp(start, end, t);
    }
}

/*
 * Geometry operations
 */
static inline int s3d_clip_against_plane(s3d_vec_t plane,
                                         s3d_vec_t norm,
                                         s3d_triangle_t in,
                                         s3d_triangle_t out[2])
{
    norm = s3d_vec_norm(norm);
    s3d_fixed_t plane_d = s3d_vec_dot(norm, plane);
    s3d_vec_t *inside[3];
    int inside_count = 0;
    s3d_vec_t *outside[3];
    int outside_count = 0;
    s3d_fixed_t d0 = s3d_vec_dot(in.p[0], norm) - plane_d;
    s3d_fixed_t d1 = s3d_vec_dot(in.p[1], norm) - plane_d;
    s3d_fixed_t d2 = s3d_vec_dot(in.p[2], norm) - plane_d;
    if (d0 >= 0)
        inside[inside_count++] = &in.p[0];
    else
        outside[outside_count++] = &in.p[0];
    if (d1 >= 0)
        inside[inside_count++] = &in.p[1];
    else
        outside[outside_count++] = &in.p[1];
    if (d2 >= 0)
        inside[inside_count++] = &in.p[2];
    else
        outside[outside_count++] = &in.p[2];
    if (inside_count == 3) {
        out[0] = in;
        return 1;
    } else if (inside_count == 1 && outside_count == 2) {
        out[0].p[0] = *inside[0];
        out[0].p[1] =
            s3d_intersect_plane(norm, plane_d, *inside[0], *outside[0]);
        out[0].p[2] =
            s3d_intersect_plane(norm, plane_d, *inside[0], *outside[1]);
        return 1;
    } else if (inside_count == 2 && outside_count == 1) {
        out[0].p[0] = *inside[0];
        out[0].p[1] = *inside[1];
        out[0].p[2] =
            s3d_intersect_plane(norm, plane_d, *inside[0], *outside[0]);
        out[1].p[0] = *inside[1];
        out[1].p[1] = out[0].p[2];
        out[1].p[2] =
            s3d_intersect_plane(norm, plane_d, *inside[1], *outside[0]);
        return 2;
    }
    return 0;
}
#endif /* S3D_MATH_H */