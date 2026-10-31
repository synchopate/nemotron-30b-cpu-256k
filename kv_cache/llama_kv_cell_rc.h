#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROUTECRACK_KV_HEAD_DIM 128
#define ROUTECRACK_KV_BLOCK_SIZE 32
#define ROUTECRACK_PRECISION_Q4_0 0
#define ROUTECRACK_PRECISION_FP16 1

/**
 * struct llama_kv_cell_rc
 * 
 * Dual-format asymmetric memory cell for RouteCrack-KV:
 * - High-Entropy / Semantic Anchor tokens are stored in FP16 (256 bytes per head).
 * - Low-Entropy / Syntactic Boilerplate tokens are stored in Q4_0 (64 bytes + scale per head).
 * - Preserves Virtual Positional Index (vpos) for uncompressed RoPE phase coherence.
 */
#pragma pack(push, 4)
struct llama_kv_cell_rc {
    uint32_t vpos;            // Virtual Position Coordinate (original absolute m in context)
    uint8_t  precision;       // 0 = Q4_0, 1 = FP16 (Lookahead Guard / Exploit Primitives)
    uint8_t  flags;           // Bit 0: Attention Sink, Bit 1: L3 Pin, Bit 2: Shared Page
    uint16_t token_id;        // Original Token ID
    float    k_scale;         // Block scale factor for Q4_0 dequantization
    float    v_scale;         // Block scale factor for Q4_0 dequantization

    union {
        // Full precision FP16 storage (when precision == ROUTECRACK_PRECISION_FP16)
        struct {
            uint16_t k_fp16[ROUTECRACK_KV_HEAD_DIM]; // 256 bytes
            uint16_t v_fp16[ROUTECRACK_KV_HEAD_DIM]; // 256 bytes
        } fp16_data;

        // Compressed Q4_0 storage (when precision == ROUTECRACK_PRECISION_Q4_0)
        struct {
            uint8_t k_q4[ROUTECRACK_KV_HEAD_DIM / 2]; // 64 bytes (packed nibbles)
            uint8_t v_q4[ROUTECRACK_KV_HEAD_DIM / 2]; // 64 bytes (packed nibbles)
            uint8_t _padding[384];                    // Pad to match union stride
        } q4_data;
    } storage;
};
#pragma pack(pop)

/**
 * struct harp_page_descriptor
 * POSIX Shared Memory (/dev/shm) page descriptor for 32-token immutable prefixes.
 */
struct harp_page_descriptor {
    uint64_t page_hash;       // 64-bit FNV-1a Hash of the 32-token sequence
    uint32_t page_id;         // Global slot ID in shared memory
    uint16_t token_count;     // Number of valid tokens (<= 32)
    uint16_t ref_count;       // Active consumer counter
    uint64_t shm_byte_offset; // Absolute byte offset in /dev/shm arena
};

#ifdef __cplusplus
}
#endif
