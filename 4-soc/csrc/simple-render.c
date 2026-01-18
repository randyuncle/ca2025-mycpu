#include <stdint.h>
#include <stdbool.h>

#include "mmio.h"
#include "b3d-assets.h"
#include "math-sup.h" /* Supports some basic math operations in fixed-point 
                       * In here, however, some function would be overwrite into simpler version
                       */

/* Magic value to signal test completion to simulator */
#define TEST_DONE_MAGIC 0xCAFEF00Du

#define FRAME_SIZE VGA_FRAME_SIZE
#define FRAME_COUNT VGA_NUM_FRAMES
#define PIXELS_PER_WORD VGA_PIXELS_PER_WORD
#define WORDS_PER_FRAME VGA_WORDS_PER_FRAME
#define PALETTE_SIZE 10  // cube color count
#define PALETTE_MAX 16   // VGA palette entries

typedef struct {
    int32_t *triangles;   /* Triangle vertices: 9 Q15.16 fixed-point per tri (ax,ay,az,...) */
    int32_t triangle_count; /* Number of triangles */
    int32_t vertex_count;   /* Total vertex components (triangle_count * 9) */
} s3d_mesh_t;

/* Depth buffer type (take 32-bit as default) */
typedef int32_t s3d_depth_t;

/* Global data */
static int32_t zbuffer[FRAME_SIZE];        // Z-buffer
static uint8_t frame_buffer[FRAME_SIZE];       // Current frame buffer
static s3d_mesh_t mesh;

/* 3D point/vertex */
typedef struct {
    int32_t x, y, z;
} s3d_point_t;

/* Triangle defined by 3 vertices */
typedef struct {
    s3d_point_t v[3];
} s3d_tri_t;

/* Camera parameters */
typedef struct {
    int32_t x, y, z;          /* position */
    int32_t yaw, pitch, roll; /* orientation in radians */
} s3d_camera_t;

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

/* color of cube (downsampling from the original color declaration in `cube.c`) */
static const uint8_t cube_palette[12] = {
    0x3e, 0x2a, 0x19, 0x15, 0x10, 0x3e, 0x3f, 0x3e, 0x39, 0x24, 0x17, 0x1b
};

/* The color that would be used in current pallete */
static const uint8_t render_pallete[10] = {
    0x3e, 0x2a, 0x19, 0x15, 0x10, 0x3f, 0x39, 0x24, 0x17, 0x1b
};

// Initialize VGA palette with cube colors
void vga_init_palette(void)
{
    for (int i = 0; i < PALETTE_SIZE; i++) {
        vga_write32(VGA_ADDR_PALETTE(i), render_pallete[i] & 0x3F);
    }
    // Fill remaining palette entries with black
    for (int i = PALETTE_SIZE; i < PALETTE_MAX; i++) {
        vga_write32(VGA_ADDR_PALETTE(i), 0x00);
    }
}

/* back-face culling that applies after the division */
static bool backface_cull(s3d_vec_t *a, s3d_vec_t *b, s3d_vec_t *c)
{
    s3d_fixed_t abx = b->x - a->x;
    s3d_fixed_t aby = b->y - a->y;
    s3d_fixed_t acx = c->x - a->x;
    s3d_fixed_t acy = c->y - a->y;

    s3d_fixed_t cross = S3D_FP_MUL(abx, acy) - S3D_FP_MUL(aby, acx);
    return cross <= 0;
}

typedef struct {
    s3d_vec_t v[8]; /* up to 8 vertices after clipping */
    int count;
} s3d_poly_t;

static int inside_plane(s3d_vec_t *v, int plane)
{
    switch (plane) {
        case 0: return v->x >= -v->w; /* left */
        case 1: return v->x <=  v->w; /* right */
        case 2: return v->y >= -v->w; /* bottom */
        case 3: return v->y <=  v->w; /* top */
        case 4: return v->z >= 0;         /* near */
        case 5: return v->z <= v->w;  /* far */
    }
    return 0;
}

static inline s3d_scalar_t plane_dist(s3d_vec_t v, int plane)
{
    switch(plane) {
        case 0: return v.x + S3D_FP_DIV(S3D_FP_ONE, v.w); /* x + w */
        case 1: return S3D_FP_DIV(S3D_FP_ONE, v.w) - v.x; /* w - x */
        case 2: return v.y + S3D_FP_DIV(S3D_FP_ONE, v.w);
        case 3: return S3D_FP_DIV(S3D_FP_ONE, v.w) - v.y;
        case 4: return v.z;                     /* near */
        case 5: return S3D_FP_DIV(S3D_FP_ONE, v.w) - v.z;    
        default: return 0;
    }
}

/* overwritten intersact function for simple render */
static s3d_vec_t intersect(s3d_vec_t a, s3d_vec_t b, int plane)
{
    s3d_fixed_t da = plane_dist(a, plane);
    s3d_fixed_t db = plane_dist(b, plane);
    s3d_fixed_t denom = S3D_FP_SUB(da, db);
    if (denom < S3D_FP_EPSILON)
        denom = 1;

    s3d_fixed_t t = S3D_FP_DIV(da, denom);

    /* clamp to [0, S3D_FP_ONE] */
    t = s3d_fp_min(s3d_fp_max(t, 0), S3D_FP_ONE);

    return s3d_vec_lerp(a, b, t);
}

static void clip_plane(s3d_poly_t *in, s3d_poly_t *out, int plane)
{
    out->count = 0;
    s3d_vec_t *a = &in->v[(in->count) - 1];
    int a_in = inside_plane(a, plane);

    for (int i = 0; i < in->count; i++) {
        s3d_vec_t *b = &in->v[i];
        int b_in = inside_plane(b, plane);

        if (a_in && b_in) {
            out->v[out->count++] = *b;
        } else if (a_in && !b_in) {
            /* exiting */
            out->v[out->count++] = intersect(*a, *b, plane);
        } else if (!a_in && b_in) {
            /* entering */
            out->v[out->count++] = intersect(*a, *b, plane);
            out->v[out->count++] = *b;
        }

        a = b;
        a_in = b_in;
    }
}

static int clip_triangle(s3d_vec_t *v0, s3d_vec_t *v1, s3d_vec_t *v2, s3d_poly_t *out)
{
    s3d_poly_t tmp1, tmp2;
    tmp1.count = 3;
    tmp1.v[0] = *v0;
    tmp1.v[1] = *v1;
    tmp1.v[2] = *v2;

    /* Clip for six planes */
    for (int p = 0; p < 6; p++) {
        clip_plane(&tmp1, &tmp2, p);
        if (tmp2.count < 3)
            return 0;
        tmp1 = tmp2;
    }

    *out = tmp1;
    return 1;
}

/* Clamp integer to [lo, hi] range - inline function avoids double evaluation */
static inline int s3d_clamp_int(int v, int lo, int hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

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

#define PUT_PIXEL(i)                                                 \
    do {                                                             \
        s3d_scalar_t d =                                             \
            S3D_DEPTH_OFFSET - S3D_FP_MUL(w, S3D_DEPTH_SCALE);       \
        if (d < 0)                                                   \
            d = 0;                                                   \
        else if (d > S3D_FP_ONE)                                     \
            d = S3D_FP_ONE;                                          \
        if (d < dp[i]) {                                             \
            dp[i] = d;                                               \
            pp[i] = c;                                               \
        }                                                            \
        w = S3D_FP_ADD(w, w_step);                                   \
    } while (0)

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

    size_t s3d_width_pot = 8;
    /* Initialize row base for iterative update (addition vs multiplication) */
    size_t row_base = (size_t) y_start << (size_t) s3d_width_pot;

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

static void s3d_triangle(const s3d_tri_t *tri, uint8_t c)
{
    s3d_triangle_t t = (s3d_triangle_t){{
        {tri->v[0].x, tri->v[0].y, tri->v[0].z, S3D_FP_ONE},
        {tri->v[1].x, tri->v[1].y, tri->v[1].z, S3D_FP_ONE},
        {tri->v[2].x, tri->v[2].y, tri->v[2].z, S3D_FP_ONE},
    }};

    TRANSFORM_TRI(t, s3d_model);

    /* simple clipping (without considering near and far) */
    s3d_poly_t *out;
    if (clip_triangle(&t.p[0], &t.p[1], &t.p[2], out) == 0)
        return;

    /* Call raterization based on the clipping result */
    if (out->count >= 3) {
        for (int i = 1; (i + 1) < out->count; i++) {
            s3d_triangle_t local = (s3d_triangle_t){{
                {out->v[0].x, out->v[0].y, out->v[0].z, out->v[0].w},
                {out->v[i].x, tri->v[i].y, tri->v[i].z, out->v[i].w},
                {out->v[i + 1].x, out->v[i + 1].y, out->v[i + 1].z, out->v[i + 1].w},
            }};

            TRANSFORM_TRI(t, s3d_proj);

            /* Save 1/w before perspective divide for perspective-correct depth.
            * After projection, w = z_view. We store 1/w in z component so it
            * interpolates correctly during screen-space clipping.
            */
            s3d_fixed_t w_inv0 = S3D_FP_DIV(S3D_FP_ONE, local.p[0].w);
            s3d_fixed_t w_inv1 = S3D_FP_DIV(S3D_FP_ONE, local.p[1].w);
            s3d_fixed_t w_inv2 = S3D_FP_DIV(S3D_FP_ONE, local.p[2].w);

            PERSPECTIVE_DIV(local);

            /* Store 1/w in z for perspective-correct depth interpolation */
            local.p[0].z = w_inv0;
            local.p[1].z = w_inv1;
            local.p[2].z = w_inv2;

            if (backface_cull(&local.p[0], &local.p[1], &local.p[2]))
                continue;

            s3d_fixed_t xs = S3D_FP_MUL(S3D_INT_TO_FP(s3d_width) , S3D_FP_HALF);
            s3d_fixed_t ys = S3D_FP_MUL(S3D_INT_TO_FP(s3d_height), S3D_FP_HALF);

            NDC_TO_SCREEN(local.p[0], xs, ys);
            NDC_TO_SCREEN(local.p[1], xs, ys);
            NDC_TO_SCREEN(local.p[2], xs, ys);
            
            raster_vertex_t rv[3] = {
                {local.p[0].x, local.p[0].y, local.p[0].z},
                {local.p[1].x, local.p[1].y, local.p[1].z},
                {local.p[2].x, local.p[2].y, local.p[2].z},
            }; // the vertices are fixed-point in default

            s3d_rasterize(rv, c);
        }
    }
}


/* Camera setting function adapted from `b3d.c` */
void s3d_set_camera(const s3d_camera_t *cam)
{
    if (!cam)  
        return;
    // Set camera position and orientation here
    s3d_camera_params = *cam;
    s3d_camera = (s3d_vec_t) {cam->x, cam->y, cam->z, S3D_FP_ONE};
    // s3d_vec_t up = {0, 1, 0, 1};
    // s3d_vec_t target = {0, 0, 1, 1};
    // up = s3d_mat_mul_vec(s3d_mat_rot_z(cam->roll), up);
    // target = s3d_mat_mul_vec(s3d_mat_rot_x(cam->pitch), target);
    // target = s3d_mat_mul_vec(s3d_mat_rot_y(cam->yaw), target);
    // target = s3d_vec_add(s3d_camera, target);
    // s3d_view = s3d_mat_qinv(s3d_mat_point_at(s3d_camera, target, up));
}

static int32_t min_y, max_y, max_xz;
static int32_t y_offset, z_offset;

static inline void s3d_mesh_bounds(const s3d_mesh_t *mesh,
                                   int32_t *min_y,
                                   int32_t *max_y,
                                   int32_t *max_xz)
{
    if (!mesh || !mesh->triangles || mesh->vertex_count < 3) {
        if (min_y)
            *min_y = 0;
        if (max_y)
            *max_y = 0;
        if (max_xz)
            *max_xz = 0;
        return;
    }

    int32_t miny = mesh->triangles[1], maxy = mesh->triangles[1];
    int32_t maxxz = 0;

    for (int i = 0; i < mesh->vertex_count; i += 3) {
        int32_t x = mesh->triangles[i + 0];
        int32_t y = mesh->triangles[i + 1];
        int32_t z = mesh->triangles[i + 2];
        int32_t ax = (x < 0) ? -x : x;
        int32_t az = (z < 0) ? -z : z;

        if (y < miny)
            miny = y;
        if (y > maxy)
            maxy = y;
        if (ax > maxxz)
            maxxz = ax;
        if (az > maxxz)
            maxxz = az;
    }

    if (min_y)
        *min_y = miny;
    if (max_y)
        *max_y = maxy;
    if (max_xz)
        *max_xz = maxxz;
}

/* Set the cube mesh (single mesh settings only) */
static int s3d_set_mesh()
{
    mesh.triangle_count = cube[0];
    mesh.vertex_count = cube[1];
    mesh.triangles = (int32_t *)&cube[2];
    s3d_mesh_bounds(&mesh, &min_y, &max_y, &max_xz);
    y_offset = S3D_FP_DIV(S3D_FP_ADD(min_y, max_y), S3D_INT_TO_FP(2));
    z_offset = -S3D_FP_ADD(S3D_FP_SUB(max_y, min_y), max_xz);
    if (z_offset > -S3D_INT_TO_FP(1))
        z_offset = -S3D_INT_TO_FP(1); /* Minimum distance */
    s3d_set_camera(&(s3d_camera_t) {0, y_offset, z_offset, 0, 0, 0});
    return 0;
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
               int w, int h /*, int32_t fov*/)
{
    if (!pixel_buffer || !depth_buffer || w <= 0 || h <= 0)
        return false;

    s3d_pixels = pixel_buffer;
    s3d_depth = depth_buffer;
    s3d_width = w;
    s3d_height = h;
    s3d_proj = s3d_mat_proj(0,
                            S3D_FP_DIV(S3D_INT_TO_FP(s3d_width),
                                        S3D_INT_TO_FP(s3d_height)),
                            S3D_NEAR_DISTANCE,
                            S3D_FAR_DISTANCE);
    s3d_set_camera(&(s3d_camera_t) {0, 0, 0, 0, 0, 0});
    return true;
}

void s3d_clear()
{
    if (!s3d_depth || !s3d_pixels || s3d_width <= 0 || s3d_height <= 0)
        return;

    size_t count = FRAME_SIZE;

    for (size_t i = 0; i < count; ++i) {
        s3d_depth[i] = S3D_DEPTH_CLEAR;
        s3d_pixels[i] = 0; /* Black */
    }
}

static void render_mesh(int frame)
{
    s3d_clear();
    s3d_reset();

    s3d_fixed_t angle = S3D_FP_MUL(S3D_INT_TO_FP(frame), S3D_FP_DIV(S3D_FP_2PI, S3D_INT_TO_FP(FRAME_COUNT)));
    s3d_rotate_y(angle);

    for (int t = 0, tv = 0; t < mesh.triangle_count && tv < mesh.vertex_count; t++, tv += 9) {
        s3d_triangle(
            &(s3d_tri_t) {
                {
                    {mesh.triangles[tv], mesh.triangles[tv + 1], mesh.triangles[tv + 2]},
                    {mesh.triangles[tv + 3], mesh.triangles[tv + 4], mesh.triangles[tv + 5]},
                    {mesh.triangles[tv + 6], mesh.triangles[tv + 7], mesh.triangles[tv + 8]}
                }
            },
            cube_palette[t]
        );
    }
}

// Simple delay function (~20Hz frame rate)
// Use inline assembly to prevent compiler optimization
static inline void delay(uint32_t cycles)
{
    for (uint32_t i = 0; i < cycles; i++)
        __asm__ volatile("nop");
}

int main () 
{
    // Verify VGA peripheral presence
    uint32_t id = vga_read32(VGA_ADDR_ID);
    if (id != VGA_EXPECTED_ID)
        return 1;

    // Initialize palette and enable display
    vga_init_palette();
    vga_write32(VGA_ADDR_CTRL, 0x01);

    s3d_init(frame_buffer, zbuffer, VGA_FRAME_WIDTH, VGA_FRAME_HEIGHT);
    if (s3d_set_mesh() != 0)
        return 1;

    vga_write32(VGA_ADDR_UPLOAD_ADDR,
                ((uint32_t) (0 & 0xF) << 16) | 0);
    // Render cube mesh with 12 angles
    for (int frame = 0; frame < FRAME_COUNT; frame++) {
        // render the current mesh
        render_mesh(frame);
        vga_write32(VGA_ADDR_CTRL, 0x01);
        vga_write32(VGA_ADDR_UPLOAD_ADDR,
                    ((uint32_t) (0 & 0xF) << 16) | 0);
        // Upload frame to VGA
        for (int i = 0; i < FRAME_SIZE; i += PIXELS_PER_WORD) {
            uint32_t packed = vga_pack8_pixels(&frame_buffer[i]);
            vga_write32(VGA_ADDR_STREAM_DATA, packed);
        }
        vga_write32(VGA_ADDR_CTRL, (0 << 4) | 0x01);
        delay(50000);
    }

    // for (uint32_t frame = 0;;) {
    //     vga_write32(VGA_ADDR_CTRL, (frame << 4) | 0x01);
    //     delay(50000);
    //     frame = (frame + 1 < FRAME_COUNT) ? frame + 1 : 0;
    // }
    return 0;
}