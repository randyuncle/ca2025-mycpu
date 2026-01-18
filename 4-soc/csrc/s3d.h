#ifndef S3D_H
#define S3D_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Matrix stack size for push/pop operations */
#define S3D_MATRIX_STACK_SIZE 16
#define VGA_WIDTH_POWER_OF_2 8

/* Depth buffer type (take 32-bit as default) */
typedef int32_t s3d_depth_t;

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

/* Public API functions */
bool s3d_init(uint8_t *pixel_buffer, s3d_depth_t *depth_buffer,
               int w, int h, int32_t fov);
void s3d_clear(int area);
void s3d_reset(void);
void s3d_rotate_y(int32_t angle);
void s3d_set_camera(const s3d_camera_t *cam);
bool s3d_triangle(const s3d_tri_t *tri, uint8_t c);

#endif /* S3D_H */