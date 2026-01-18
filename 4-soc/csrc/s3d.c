/* 
 * simple 3D rendering library for VGA test 
 * Most of the code are derived from B3D library with only focusing on
 * fixed-point arithmetic through all of the functions.
 */

#include <stdbool.h>

#include "math-sup.h"
#include "s3d.h"

/* Global state */
/* Basic elements */
int s3d_width, s3d_height; // expected to be VGA_WIDTH, VGA_HEIGHT
uint8_t *s3d_pixels; // framebuffer pointer
s3d_depth_t *s3d_depth; // z-buffer pointer

/* Transformation matrices and camera */
static s3d_mat_t s3d_model, s3d_view, s3d_proj;
static s3d_vec_t s3d_camera;
static s3d_camera_t s3d_camera_params;
static int32_t s3d_fov_degrees;          /* Current FOV for queries */

/* Matrix stack for push/pop operations */
// static s3d_mat_t s3d_matrix_stack[S3D_MATRIX_STACK_SIZE];
static int s3d_matrix_stack_top = 0;

/* Debug counters */
static size_t s3d_clip_drop_count = 0;

/* Perspective-correct depth conversion constants.
 * Converts interpolated 1/w to normalized depth [0, 1].
 * Formula: depth = S3D_DEPTH_OFFSET - w_inv * S3D_DEPTH_SCALE
 *
 * Fixed-point considerations:
 * - w_inv ranges from 1/far to 1/near (e.g., 0.01 to 10 for default planes)
 * - In Q15.16: 10 = 655360, 0.01 = 655 - both fit comfortably
 * - S3D_FP_MUL uses int64_t intermediate to prevent overflow
 * - Result depth is in [0, S3D_FP_ONE] range
 *
 * Precision note: In Q15.16, near-plane depth may be ~0.0001 instead of
 * exact 0 due to rounding in constant conversion. This is expected and
 * does not affect z-buffer correctness since depth ordering is preserved.
 */
#define S3D_DEPTH_OFFSET \
    (S3D_FP_DIV(S3D_FAR_DISTANCE, (S3D_FAR_DISTANCE - S3D_NEAR_DISTANCE)))
#define S3D_DEPTH_SCALE                       \
    (S3D_FP_DIV((S3D_FP_MUL(S3D_NEAR_DISTANCE, S3D_FAR_DISTANCE)), \
     (S3D_FAR_DISTANCE - S3D_NEAR_DISTANCE)))

/* Precomputed depth constants in scalar format (initialized on first use) */
static s3d_scalar_t s3d_depth_offset_fp = 0;
static s3d_scalar_t s3d_depth_scale_fp = 0;
static bool s3d_depth_constants_valid = false;

static inline void s3d_init_depth_constants(void)
{
    if (s3d_depth_constants_valid)
        return;
    s3d_depth_offset_fp = S3D_DEPTH_OFFSET;
    s3d_depth_scale_fp = S3D_DEPTH_SCALE;
    s3d_depth_constants_valid = true;
}

/* Cached screen-space clipping planes (updated when resolution changes) */
static s3d_vec_t s3d_screen_planes[4][2];
static int s3d_planes_cached_w = 0, s3d_planes_cached_h = 0;

static void s3d_update_screen_planes(void)
{
    if (s3d_planes_cached_w == s3d_width && s3d_planes_cached_h == s3d_height)
        return;

    /* Top edge */
    s3d_screen_planes[0][0] = (s3d_vec_t) {0, S3D_FP_HALF, 0, S3D_FP_ONE};
    s3d_screen_planes[0][1] = (s3d_vec_t) {0, S3D_FP_ONE, 0, S3D_FP_ONE};
    /* Bottom edge */
    s3d_screen_planes[1][0] = (s3d_vec_t) {0, S3D_INT_TO_FP(s3d_height), 0, S3D_FP_ONE};
    s3d_screen_planes[1][1] = (s3d_vec_t) {0, -S3D_FP_ONE, 0, S3D_FP_ONE};
    /* Left edge */
    s3d_screen_planes[2][0] = (s3d_vec_t) {S3D_FP_HALF, 0, 0, S3D_FP_ONE};
    s3d_screen_planes[2][1] = (s3d_vec_t) {S3D_FP_ONE, 0, 0, S3D_FP_ONE};
    /* Right edge */
    s3d_screen_planes[3][0] = (s3d_vec_t) {S3D_INT_TO_FP(s3d_width), 0, 0, S3D_FP_ONE};
    s3d_screen_planes[3][1] = (s3d_vec_t) {-S3D_FP_ONE, 0, 0, S3D_FP_ONE};
    s3d_planes_cached_w = s3d_width;
    s3d_planes_cached_h = s3d_height;
}

static inline s3d_scalar_t s3d_depth_load(s3d_depth_t v)
{
    return (s3d_scalar_t) v;
}

static inline s3d_depth_t s3d_depth_store(s3d_scalar_t v)
{
    return (s3d_depth_t) v;
}

/* Swap span endpoints (x and z) when start > end */
#define SWAP_SPAN(sx, sz, ex, ez, tmp) \
    do {                               \
        tmp = sx;                      \
        sx = ex;                       \
        ex = tmp;                      \
        tmp = sz;                      \
        sz = ez;                       \
        ez = tmp;                      \
    } while (0)

#define TRANSFORM_TRI(tri, mat)                        \
    do {                                               \
        (tri).p[0] = s3d_mat_mul_vec(mat, (tri).p[0]); \
        (tri).p[1] = s3d_mat_mul_vec(mat, (tri).p[1]); \
        (tri).p[2] = s3d_mat_mul_vec(mat, (tri).p[2]); \
    } while (0)

/* Perspective divide for all 3 vertices */
#define PERSPECTIVE_DIV(tri)                                \
    do {                                                    \
        (tri).p[0] = s3d_vec_div((tri).p[0], (tri).p[0].w); \
        (tri).p[1] = s3d_vec_div((tri).p[1], (tri).p[1].w); \
        (tri).p[2] = s3d_vec_div((tri).p[2], (tri).p[2].w); \
    } while (0)

/* Convert NDC vertex to screen coordinates */
#define NDC_TO_SCREEN(p, xs, ys)     \
    do {                             \
        (p).x = S3D_FP_MUL((S3D_FP_ADD((p).x, S3D_FP_ONE)), (xs));  \
        (p).y = S3D_FP_MUL((S3D_FP_ADD((p).y, S3D_FP_ONE)), (ys)); \
    } while (0)

/* Clamp integer to [lo, hi] range - inline function avoids double evaluation */
static inline int s3d_clamp_int(int v, int lo, int hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

/* Pixel write macro for scanline unrolling.
 * Uses perspective-correct depth: depth = offset - w_inv * scale.
 * Expects: w = current 1/w, w_step = delta 1/w per pixel.
 * Uses cached s3d_depth_offset_fp and s3d_depth_scale_fp.
 * Clamps depth to [0, S3D_FP_ONE] for numerical stability.
 */
#define PUT_PIXEL(i)                                                 \
    do {                                                             \
        s3d_scalar_t d =                                             \
            s3d_depth_offset_fp - S3D_FP_MUL(w, s3d_depth_scale_fp); \
        if (d < 0)                                                   \
            d = 0;                                                   \
        else if (d > S3D_FP_ONE)                                     \
            d = S3D_FP_ONE;                                          \
        if (d < s3d_depth_load(dp[i])) {                             \
            dp[i] = s3d_depth_store(d);                              \
            pp[i] = c;                                               \
        }                                                            \
        w = S3D_FP_ADD(w, w_step);                                   \
    } while (0)

/* Inner API functions (most of the functions considered fixed-point calc.) */

/* Edge interpolation state for rasterizer.
 * Uses 1/w for perspective-correct depth interpolation.
 */
typedef struct {
    s3d_scalar_t x, w_inv;   /* start position: x coord and 1/w */
    s3d_scalar_t dx, dw_inv; /* delta per scanline */
    s3d_scalar_t t;          /* interpolation parameter [0, 1] */
    s3d_scalar_t t_step;     /* step per scanline */
} raster_edge_t;

static void raster_half(int y_start,
                          int y_end,
                          raster_edge_t *left,
                          raster_edge_t *right,
                          uint8_t c)
{
    s3d_scalar_t tmp = 0;

    /* Clamp y range to screen bounds, fast-forward edge parameters.
     * Fixed-point: use int64_t to prevent overflow when t_step * skip
     * could exceed INT32_MAX on zoomed-in geometry.
     * Float: standard multiplication is safe.
     */
    if (y_start < 0) {
        s3d_scalar_t skip = -S3D_INT_TO_FP(y_start);
        left->t += (s3d_scalar_t) S3D_FP_MUL(left->t_step, skip);
        right->t += (s3d_scalar_t) S3D_FP_MUL(right->t_step, skip);
        y_start = 0;
    }
    if (y_end > s3d_height)
        y_end = s3d_height;
    if (y_start >= y_end)
        return;

    /* Initialize row base for iterative update (addition vs multiplication) */
    size_t row_base = (size_t) (((size_t) y_start) << VGA_WIDTH_POWER_OF_2);

    for (int y = y_start; y < y_end; y++) {
        /* Interpolate x and 1/w along edges */
        s3d_scalar_t sx = left->x + S3D_FP_MUL(left->dx, left->t);
        s3d_scalar_t sw = left->w_inv + S3D_FP_MUL(left->dw_inv, left->t);
        s3d_scalar_t ex = right->x + S3D_FP_MUL(right->dx, right->t);
        s3d_scalar_t ew = right->w_inv + S3D_FP_MUL(right->dw_inv, right->t);
        if (sx > ex)
            SWAP_SPAN(sx, sw, ex, ew, tmp);
        s3d_scalar_t dx = ex - sx;
        if (dx < S3D_DEGEN_THRESHOLD) {
            left->t += left->t_step;
            right->t += right->t_step;
            row_base += s3d_width;
            continue;
        }

        /*
         * Depth interpolation: skip spans with
         * large depth range - skip these degenerate cases.
         * Use int64_t to avoid overflow in the comparison itself.
         */
        s3d_scalar_t dw = ew - sw;
        s3d_scalar_t dw_abs = dw < 0 ? -dw : dw;
        /* Max ratio ~32 ensures w_step fits in int32_t with margin */
        if (dw_abs > S3D_FP_MUL(dx, S3D_INT_TO_FP(32))) {
            left->t += left->t_step;
            right->t += right->t_step;
            row_base += s3d_width;
            continue;
        }
        s3d_scalar_t w_step = S3D_FP_DIV(dw, dx);

        int start = S3D_FP_TO_INT(sx), end = S3D_FP_TO_INT(ex);
        start = s3d_clamp_int(start, 0, s3d_width);
        end = s3d_clamp_int(end, 0, s3d_width);
        if (start >= end) {
            left->t += left->t_step;
            right->t += right->t_step;
            row_base += s3d_width;
            continue;
        }

        /* Compute starting 1/w at first visible pixel */
        s3d_scalar_t w = sw + S3D_FP_MUL(w_step, S3D_INT_TO_FP(start) - sx);

        s3d_depth_t *dp = s3d_depth + row_base + start;
        uint8_t *pp = s3d_pixels + row_base + start;
        int n = end - start;
        while (n >= 4) {
            PUT_PIXEL(0);
            PUT_PIXEL(1);
            PUT_PIXEL(2);
            PUT_PIXEL(3);
            dp += 4, pp += 4;
            n -= 4;
        }
        while (n > 0) {
            PUT_PIXEL(0);
            dp += 1, pp += 1;
            n -= 1;
        }

        left->t += left->t_step;
        right->t += right->t_step;
        row_base += s3d_width;
    }
}

/* Screen-space vertex for rasterization.
 * @x: screen-space x coordinate
 * @y: screen-space y coordinate
 * @w_inv: 1/w for perspective-correct depth interpolation (w = z_view)
 */
typedef struct {
    s3d_scalar_t x, y, w_inv;
} raster_vertex_t;

static void s3d_rasterize(const raster_vertex_t v[3], uint8_t c)
{
    raster_vertex_t a = {S3D_FP_FLOOR(v[0].x), S3D_FP_FLOOR(v[0].y),
                         v[0].w_inv};
    raster_vertex_t b = {S3D_FP_FLOOR(v[1].x), S3D_FP_FLOOR(v[1].y),
                         v[1].w_inv};
    raster_vertex_t cv = {S3D_FP_FLOOR(v[2].x), S3D_FP_FLOOR(v[2].y),
                          v[2].w_inv};
    
    s3d_scalar_t min_x = s3d_fp_min(s3d_fp_min(a.x, b.x), cv.x);
    s3d_scalar_t max_x = s3d_fp_max(s3d_fp_max(a.x, b.x), cv.x);
    s3d_scalar_t min_y = s3d_fp_min(s3d_fp_min(a.y, b.y), cv.y);
    s3d_scalar_t max_y = s3d_fp_max(s3d_fp_max(a.y, b.y), cv.y);
    if (max_x < 0 || min_x >= S3D_INT_TO_FP(s3d_width) ||
        max_y < 0 || min_y >= S3D_INT_TO_FP(s3d_height)) {
        return; /* Outside screen bounds */
    }

    /* Sort vertices by Y coordinate (bubble sort for 3 elements) */
    raster_vertex_t tmp;
    if (a.y > b.y) {
        tmp = a;
        a = b;
        b = tmp;
    }
    if (a.y > cv.y) {
        tmp = a;
        a = cv;
        cv = tmp;
    }
    if (b.y > cv.y) {
        tmp = b;
        b = cv;
        cv = tmp;
    }

    /* Guard against degenerate triangles (division by zero) */
    s3d_scalar_t dy_total = cv.y - a.y;
    s3d_scalar_t dy_top = b.y - a.y;
    if (dy_total < S3D_DEGEN_THRESHOLD)
        return;

    /* Setup left edge (A to C, spans entire triangle) */
    raster_edge_t left = {
        .x = a.x,
        .w_inv = a.w_inv,
        .dx = cv.x - a.x,
        .dw_inv = cv.w_inv - a.w_inv,
        .t = 0,
        .t_step = S3D_FP_DIV(S3D_FP_ONE, dy_total),
    };

    /* Setup right edge for top half (A to B) */
    raster_edge_t right = {
        .x = a.x,
        .w_inv = a.w_inv,
        .dx = b.x - a.x,
        .dw_inv = b.w_inv - a.w_inv,
        .t = 0,
        .t_step = (dy_top > S3D_DEGEN_THRESHOLD)
                      ? S3D_FP_DIV(S3D_FP_ONE, dy_top)
                      : 0,
    };
    /* Rasterize top half: right edge from A toward B */
    raster_half(S3D_FP_TO_INT(a.y), S3D_FP_TO_INT(b.y), &left, &right, c);

    /* Setup right edge for bottom half (B to C) */
    s3d_scalar_t dy_bot = cv.y - b.y;
    right = (raster_edge_t) {
        .x = b.x,
        .w_inv = b.w_inv,
        .dx = cv.x - b.x,
        .dw_inv = cv.w_inv - b.w_inv,
        .t = 0,
        .t_step = (dy_bot > S3D_DEGEN_THRESHOLD)
                      ? S3D_FP_DIV(S3D_FP_ONE, dy_bot)
                      : 0,
    };

    /* Rasterize bottom half: right edge from B toward C */
    raster_half(S3D_FP_TO_INT(b.y), S3D_FP_TO_INT(cv.y), &left, &right, c);
}
#undef PUT_PIXEL

/* Main API */

bool s3d_triangle(const s3d_tri_t *tri, uint8_t c)
{
    if (!tri || !s3d_pixels || !s3d_depth)
        return false;

    s3d_triangle_t t = (s3d_triangle_t){{
        {tri->v[0].x, tri->v[0].y, tri->v[0].z, S3D_FP_ONE},
        {tri->v[1].x, tri->v[1].y, tri->v[1].z, S3D_FP_ONE},
        {tri->v[2].x, tri->v[2].y, tri->v[2].z, S3D_FP_ONE},
    }};

    // I consider always accepted world-culling in this case
    TRANSFORM_TRI(t, s3d_model);
    s3d_vec_t line_a = s3d_vec_sub(t.p[1], t.p[0]);
    s3d_vec_t line_b = s3d_vec_sub(t.p[2], t.p[0]);
    s3d_vec_t normal = s3d_vec_cross(line_a, line_b);
    s3d_vec_t cam_ray = s3d_vec_sub(t.p[0], s3d_camera);
    if (s3d_vec_dot(normal, cam_ray) > S3D_CULL_THRESHOLD)
        return false;
    TRANSFORM_TRI(t, s3d_view);

    /* Near-distance setup */
    s3d_triangle_t clipped[2];
    int count = s3d_clip_against_plane((s3d_vec_t) {0, 0, S3D_NEAR_DISTANCE, S3D_FP_ONE},
                                       (s3d_vec_t) {0, 0, S3D_FP_ONE, S3D_FP_ONE}, t, clipped);
    if (count == 0)
        return false;

    /* Far-plane clipping: keep vertices where z <= S3D_FAR_DISTANCE */
    s3d_triangle_t view_clipped[4];
    int view_count = 0;
    for (int i = 0; i < count; ++i) {
        s3d_triangle_t fc[2];
        int n =
            s3d_clip_against_plane((s3d_vec_t) {0, 0, S3D_FAR_DISTANCE, S3D_FP_ONE},
                                   (s3d_vec_t) {0, 0, -S3D_FP_ONE, S3D_FP_ONE}, clipped[i], fc);
        for (int j = 0; j < n && view_count < 4; ++j)
            view_clipped[view_count++] = fc[j];
    }
    if (view_count == 0)
        return false;

    /* Clip against screen planes */
    s3d_triangle_t buf_a[S3D_CLIP_BUFFER_SIZE];
    s3d_triangle_t buf_b[S3D_CLIP_BUFFER_SIZE];
    s3d_triangle_t *src = buf_a, *dst = buf_b;
    int src_count = 0;
    for (int n = 0; n < view_count; ++n) {
        t = view_clipped[n];
        TRANSFORM_TRI(t, s3d_proj);
        if (s3d_fp_abs(t.p[0].w) < S3D_FP_EPSILON || s3d_fp_abs(t.p[1].w) < S3D_FP_EPSILON ||
            s3d_fp_abs(t.p[2].w) < S3D_FP_EPSILON)
            continue;

        /* Save 1/w before perspective divide for perspective-correct depth.
         * After projection, w = z_view. We store 1/w in z component so it
         * interpolates correctly during screen-space clipping.
         */
        s3d_fixed_t w_inv0 = S3D_FP_DIV(S3D_FP_ONE, t.p[0].w);
        s3d_fixed_t w_inv1 = S3D_FP_DIV(S3D_FP_ONE, t.p[1].w);
        s3d_fixed_t w_inv2 = S3D_FP_DIV(S3D_FP_ONE, t.p[2].w);

        PERSPECTIVE_DIV(t);

        /* Store 1/w in z for perspective-correct depth interpolation */
        t.p[0].z = w_inv0;
        t.p[1].z = w_inv1;
        t.p[2].z = w_inv2;

        s3d_fixed_t xs = S3D_FP_MUL(S3D_INT_TO_FP(s3d_width) , S3D_FP_HALF);
        s3d_fixed_t ys = S3D_FP_MUL(S3D_INT_TO_FP(s3d_height), S3D_FP_HALF);
        NDC_TO_SCREEN(t.p[0], xs, ys);
        NDC_TO_SCREEN(t.p[1], xs, ys);
        NDC_TO_SCREEN(t.p[2], xs, ys);
        if (src_count < S3D_CLIP_BUFFER_SIZE)
            src[src_count++] = t;
        else
            ++s3d_clip_drop_count;
    }

    /* Screen clipping */
    for (int p = 0; p < 4; ++p) {
        int dst_count = 0;
        for (int i = 0; i < src_count; ++i) {
            int n = s3d_clip_against_plane(s3d_screen_planes[p][0],
                                           s3d_screen_planes[p][1], src[i],
                                           clipped);
            for (int w = 0; w < n; ++w) {
                if (dst_count < S3D_CLIP_BUFFER_SIZE)
                    dst[dst_count++] = clipped[w];
                else
                    ++s3d_clip_drop_count;
            }
        }

        s3d_triangle_t *tmp = src;
        src = dst;
        dst = tmp;
        src_count = dst_count;
    }
    if (src_count == 0)
        return false;

    for (int i = 0; i < src_count; ++i) {
        raster_vertex_t rv[3] = {
            {src[i].p[0].x, src[i].p[0].y, src[i].p[0].z},
            {src[i].p[1].x, src[i].p[1].y, src[i].p[1].z},
            {src[i].p[2].x, src[i].p[2].y, src[i].p[2].z},
        }; // the vertices are fixed-point in default
        s3d_rasterize(rv, c);
    }
    return true;
}

/* Camera setting function adapted from `b3d.c` */
void s3d_set_camera(const s3d_camera_t *cam)
{
    if (!cam)  
        return;
    // Set camera position and orientation here
    s3d_camera_params = *cam;
    s3d_camera = (s3d_vec_t) {cam->x, cam->y, cam->z, S3D_FP_ONE};
    s3d_vec_t up = {0, 1, 0, 1};
    s3d_vec_t target = {0, 0, 1, 1};
    up = s3d_mat_mul_vec(s3d_mat_rot_z(cam->roll), up);
    target = s3d_mat_mul_vec(s3d_mat_rot_x(cam->pitch), target);
    target = s3d_mat_mul_vec(s3d_mat_rot_y(cam->yaw), target);
    target = s3d_vec_add(s3d_camera, target);
    s3d_view = s3d_mat_qinv(s3d_mat_point_at(s3d_camera, target, up));
}

void s3d_reset(void)
{
    s3d_model = s3d_mat_ident();
}

/* In this test, I only consider the y-axis rotation case */
void s3d_rotate_y(int32_t angle) 
{
    s3d_model = s3d_mat_mul(s3d_model, s3d_mat_rot_y(angle));
}

bool s3d_init(uint8_t *pixel_buffer, s3d_depth_t *depth_buffer,
               int w, int h, int32_t fov)
{
    if (!pixel_buffer || !depth_buffer || w <= 0 || h <= 0 || fov <= 0)
        return false;

    s3d_pixels = pixel_buffer;
    s3d_depth = depth_buffer;
    s3d_width = w;
    s3d_height = h;
    s3d_matrix_stack_top = 0;
    s3d_fov_degrees = fov;
    s3d_init_depth_constants();
    s3d_update_screen_planes();
    s3d_proj = s3d_mat_proj(fov,
                            S3D_FP_DIV(S3D_INT_TO_FP(s3d_width),
                                        S3D_INT_TO_FP(s3d_height)),
                            S3D_NEAR_DISTANCE,
                            S3D_FAR_DISTANCE);
    s3d_set_camera(&(s3d_camera_t) {0, 0, 0, 0, 0, 0});
    return true;
}

void s3d_clear(int area)
{
    if (!s3d_depth || !s3d_pixels || s3d_width <= 0 || s3d_height <= 0)
        return;

    s3d_clip_drop_count = 0;
    // /* Check for integer overflow when calculating buffer size */
    // if ((size_t)s3d_width > SIZE_MAX / (size_t)s3d_height)
    //     return; /* Prevent overflow */

    size_t count = (size_t) area;
    // /* Also check for overflow in the pixel buffer size calculation */
    // if (count > SIZE_MAX / sizeof(s3d_pixels[0]))
    //     return; /* Prevent overflow */

    for (size_t i = 0; i < count; ++i) {
        s3d_depth[i] = S3D_DEPTH_CLEAR;
        s3d_pixels[i] = 0; /* Black */
    }
}