#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define SD_API __declspec(dllexport)
#else
#define SD_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* One handle per stream. A handle must not be called concurrently.
 * Configuration: JSON {model: USD-derived model, calibration:
 * {image,left,right}}. Images are RGB uint8, with positive row strides; caller
 * owns output memory. Output is (2*width) x (height+50). Buffers must not
 * overlap. Pose: C_L[3], q_L_xyzw[4], C_R[3], q_R_xyzw[4], pitch,yaw,jaw_half
 * [rad]. Visibility order L1,L2,R1,R2: -1 unknown, 1 visible, 2 tissue, 3 self,
 * 4 other. Out-of-frame status is recomputed. Nonzero replay means ground-truth
 * replay. Return 0 on success, -1 on error; NEVER consume output after an
 * error.
 */
SD_API void *sd_create(const char *configuration_json, const char *font_path);
SD_API void sd_destroy(void *handle);
SD_API int sd_compute(void *handle, const double *values17);
SD_API int sd_render(void *handle, const double *values17, const uint8_t *left,
                     ptrdiff_t left_stride, const uint8_t *right,
                     ptrdiff_t right_stride, uint8_t *output,
                     ptrdiff_t output_stride, const int *visibility4,
                     const char *frame_id, int64_t timestamp_ns, int replay);
/* JSON valid until next inspection/destruction. Only available after successful
 * compute/render. */
SD_API const char *sd_inspect(void *handle, int detailed_geometry);
SD_API const char *sd_last_error(void);
SD_API int sd_abi_version(void);
#ifdef __cplusplus
}
#endif
