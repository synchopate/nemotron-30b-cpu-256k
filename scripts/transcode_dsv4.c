#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

extern void ggml_cpu_init(void);
extern void ggml_quantize_init(int type);

extern void dequantize_row_q8_0(const void * x, float * y, int64_t k);
extern void dequantize_row_iq2_xxs(const void * x, float * y, int64_t k);
extern void dequantize_row_iq3_xxs(const void * x, float * y, int64_t k);
extern void dequantize_row_mxfp4(const void * x, float * y, int64_t k);
extern void dequantize_row_iq1_s(const void * x, float * y, int64_t k);

extern void quantize_row_q2_0_ref(const float * x, void * y, int64_t k);
extern size_t quantize_iq1_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_iq1_m(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
extern size_t quantize_iq2_xxs(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);

static int g_initialized = 0;

void dsv4_init(void) {
    if (!g_initialized) {
        ggml_cpu_init();
        ggml_quantize_init(16); // IQ2_XXS
        ggml_quantize_init(18); // IQ3_XXS
        ggml_quantize_init(19); // IQ1_S
        ggml_quantize_init(29); // IQ1_M
        g_initialized = 1;
    }
}

// Transcode Q8_0 (type 8) -> Q2_0 Fractal-64 (type 41) in parallel across rows
void transcode_q8_0_to_q2_0(const uint8_t * src, uint8_t * dst, int64_t nrow, int64_t ncol) {
    dsv4_init();
    const int64_t src_row_bytes = (ncol / 32) * 34;
    const int64_t dst_row_bytes = (ncol / 64) * 30;
    #pragma omp parallel num_threads(32)
    {
        float * f32 = (float *)malloc(ncol * sizeof(float));
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < nrow; ++r) {
            dequantize_row_q8_0(src + r * src_row_bytes, f32, ncol);
            quantize_row_q2_0_ref(f32, dst + r * dst_row_bytes, ncol);
        }
        free(f32);
    }
}

static void dequant_expert_slice(const uint8_t * src, int src_type, float * dst, int64_t n_elem) {
    if (src_type == 16) {
        dequantize_row_iq2_xxs(src, dst, n_elem);
    } else if (src_type == 18) {
        dequantize_row_iq3_xxs(src, dst, n_elem);
    } else if (src_type == 19) {
        dequantize_row_iq1_s(src, dst, n_elem);
    } else if (src_type == 39) {
        dequantize_row_mxfp4(src, dst, n_elem);
    }
}

static int64_t expert_bytes(int ttype, int64_t n_elem) {
    if (ttype == 16) return (n_elem / 256) * 66;  // IQ2_XXS
    if (ttype == 18) return (n_elem / 256) * 98;  // IQ3_XXS
    if (ttype == 19) return (n_elem / 256) * 50;  // IQ1_S
    if (ttype == 29) return (n_elem / 256) * 56;  // IQ1_M
    if (ttype == 39) return (n_elem / 32)  * 17;  // MXFP4
    return 0;
}

// Transcode 3D MoE tensor [ncol, nrow, n_exp] with optional SwiGLU-H2 analytical importance weights
void transcode_moe_exps(
    const uint8_t * src, int src_type,
    uint8_t * dst, int dst_type,
    int64_t ncol, int64_t nrow, int64_t n_exp,
    const uint8_t * gate_src, int gate_type,
    const uint8_t * up_src, int up_type
) {
    dsv4_init();
    const int64_t n_elem = ncol * nrow;
    const int64_t src_eb = expert_bytes(src_type, n_elem);
    const int64_t dst_eb = expert_bytes(dst_type, n_elem);
    const int64_t gate_eb = (gate_src && up_src) ? expert_bytes(gate_type, n_elem) : 0;
    const int64_t up_eb   = (gate_src && up_src) ? expert_bytes(up_type,   n_elem) : 0;

    #pragma omp parallel num_threads(32)
    {
        float * f32  = (float *)malloc(n_elem * sizeof(float));
        float * fg   = (gate_src && up_src) ? (float *)malloc(n_elem * sizeof(float)) : NULL;
        float * fu   = (gate_src && up_src) ? (float *)malloc(n_elem * sizeof(float)) : NULL;
        float * qw   = (float *)malloc(ncol * sizeof(float));

        #pragma omp for schedule(dynamic, 1)
        for (int64_t e = 0; e < n_exp; ++e) {
            dequant_expert_slice(src + e * src_eb, src_type, f32, n_elem);

            if (gate_src && up_src && ncol == 2048 && nrow == 4096) {
                // Compute per-expert SwiGLU-H2 importance weights I_j = ||W_gate[j,:]||^2 * ||W_up[j,:]||^2
                dequant_expert_slice(gate_src + e * gate_eb, gate_type, fg, n_elem);
                dequant_expert_slice(up_src   + e * up_eb,   up_type,   fu, n_elem);
                double mean_qw = 0.0;
                for (int64_t j = 0; j < ncol; ++j) {
                    const float * rg = fg + j * nrow;
                    const float * ru = fu + j * nrow;
                    double sg = 0.0, su = 0.0;
                    for (int64_t k = 0; k < nrow; ++k) {
                        sg += (double)rg[k] * (double)rg[k];
                        su += (double)ru[k] * (double)ru[k];
                    }
                    qw[j] = (float)(sg * su + 1e-12);
                    mean_qw += qw[j];
                }
                mean_qw /= (double)ncol;
                float inv_mean = (mean_qw > 0.0) ? (float)(1.0 / mean_qw) : 1.0f;
                for (int64_t j = 0; j < ncol; ++j) qw[j] *= inv_mean;
            } else {
                // Use column energy + 1.0 baseline
                double mean_qw = 0.0;
                for (int64_t j = 0; j < ncol; ++j) qw[j] = 0.0f;
                for (int64_t r = 0; r < nrow; ++r) {
                    const float * row = f32 + r * ncol;
                    for (int64_t j = 0; j < ncol; ++j) {
                        qw[j] += row[j] * row[j];
                    }
                }
                for (int64_t j = 0; j < ncol; ++j) mean_qw += qw[j];
                mean_qw /= (double)ncol;
                float inv_mean = (mean_qw > 0.0) ? (float)(1.0 / mean_qw) : 1.0f;
                for (int64_t j = 0; j < ncol; ++j) qw[j] = 0.5f + 0.5f * (qw[j] * inv_mean);
            }

            uint8_t * dst_e = dst + e * dst_eb;
            if (dst_type == 16) {
                quantize_iq2_xxs(f32, dst_e, nrow, ncol, qw);
            } else if (dst_type == 29) {
                quantize_iq1_m(f32, dst_e, nrow, ncol, qw);
            } else if (dst_type == 19) {
                quantize_iq1_s(f32, dst_e, nrow, ncol, qw);
            }
        }
        free(f32);
        if (fg) free(fg);
        if (fu) free(fu);
        free(qw);
    }
}
