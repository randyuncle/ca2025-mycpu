// SPDX-License-Identifier: MIT
// MyCPU is freely redistributable under the MIT License. See the file
// "LICENSE" for information on usage and redistribution of this file.

#include <stdint.h>

#include "mmio.h"
#include "b3d-assets.h"
#include "math-sup.h"
#include "s3d.h"

/* 
 * NOTE:
 *  - Since the toolchain would not support `libc`, it is a need to 
 *    consider array-based implementations, or we manually include the
 *    mini handmade library of the required functions.
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

/* Include the tested asset from .asset section in linker list */
// extern const int32_t cube[];

/* Global data */
static int32_t zbuffer[FRAME_SIZE];        // Z-buffer
static uint8_t frame_buffer[FRAME_SIZE];       // Current frame buffer
static s3d_mesh_t mesh;

/*
 * VGA Comprehensive Test
 *
 * Tests VGA peripheral functionality including:
 * 1. Palette programming
 * 2. Framebuffer upload via streaming interface
 * 3. Display control (enable/disable, frame selection)
 *
 * Test Result Encoding:
 *   TEST_RESULT bits:
 *     [0]: Palette programming test passed
 *     [1]: Framebuffer upload test passed
 *     [2]: Display control test with s3d render and animation passed
 *   Expected: 0x7 (0b0111) = all tests passed
 *
 * Implementation Notes:
 *   - Programs a test palette and verifies by reading back
 *   - Uploads a test pattern framebuffer and verifies pixel data
 *   - Toggles display enable and frame select, verifying status
 */

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

/* real-time render the current mesh (refers to the function in `obj.c`) */
static void render_mesh(int32_t angle) 
{
    s3d_clear(FRAME_SIZE);
    s3d_reset();
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

int main(void)
{
    // Verify VGA peripheral presence
    uint32_t id = vga_read32(VGA_ADDR_ID);
    if (id != VGA_EXPECTED_ID)
        return 1;

    // Initialize palette and enable display
    vga_init_palette();
    vga_write32(VGA_ADDR_CTRL, 0x01);

    s3d_init(frame_buffer, zbuffer, VGA_FRAME_WIDTH, VGA_FRAME_HEIGHT, 70);
    if (s3d_set_mesh() != 0)
        return 1;
    // Render cube mesh with 12 angles
    for (int frame = 0; frame < FRAME_COUNT; frame++) {
        vga_write32(VGA_ADDR_UPLOAD_ADDR,
                    ((uint32_t) (frame & 0xF) << 16) | 0);
        // calculate angle in Q15.16
        int32_t angle = S3D_FP_DIV(S3D_FP_MUL(S3D_INT_TO_FP(frame), S3D_FP_ONE), S3D_INT_TO_FP(12));
        // render the current mesh
        render_mesh(angle);
        // Upload frame to VGA
        for (int i = 0; i < FRAME_SIZE; i += PIXELS_PER_WORD) {
            uint32_t packed = vga_pack8_pixels(&frame_buffer[i]);
            vga_write32(VGA_ADDR_STREAM_DATA, packed);
        }
    }

    for (uint32_t frame = 0;;) {
        vga_write32(VGA_ADDR_CTRL, (frame << 4) | 0x01);
        delay(50000);
        frame = (frame + 1 < FRAME_COUNT) ? frame + 1 : 0;
    }
}