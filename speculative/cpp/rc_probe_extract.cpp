/*
 * Extract hidden states from model for linear probe training.
 * Processes a token corpus through the model and saves (embedding, next_token) pairs.
 *
 * Output: binary file with:
 *   Header: [n_samples:u32, n_embd:u32]
 *   Per sample: [next_token:i32, embedding[n_embd]:float]
 *
 * Usage:
 *   ./rc_probe_extract -m model.gguf -c corpus.bin -o probe_data.bin [-n 10000] [-t 6]
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* corpus_path = nullptr;
    const char* output_path = "probe_data.bin";
    int max_samples = 10000;
    int n_threads = 6;
    int batch_size = 512;  // process this many tokens per batch

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i+1 < argc) corpus_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) max_samples = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
    }

    if (!model_path || !corpus_path) {
        fprintf(stderr, "Usage: %s -m model.gguf -c corpus.bin -o output.bin [-n 10000] [-t 6]\n", argv[0]);
        return 1;
    }

    // Load corpus
    FILE* cf = fopen(corpus_path, "rb");
    if (!cf) { fprintf(stderr, "Cannot open corpus: %s\n", corpus_path); return 1; }
    fseek(cf, 0, SEEK_END);
    long csz = ftell(cf);
    fseek(cf, 0, SEEK_SET);
    int n_corpus = (int)(csz / 4);
    std::vector<int32_t> corpus(n_corpus);
    fread(corpus.data(), 4, n_corpus, cf);
    fclose(cf);
    printf("Corpus: %d tokens\n", n_corpus);

    // Load model with embeddings enabled
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int n_embd = llama_model_n_embd(model);
    printf("Model loaded: n_embd=%d\n", n_embd);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = batch_size + 16;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = true;  // CRITICAL: enable embedding extraction

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // Output file
    FILE* out = fopen(output_path, "wb");
    if (!out) { fprintf(stderr, "Cannot create output: %s\n", output_path); return 1; }

    // Write placeholder header (will update n_samples at end)
    uint32_t header[2] = {0, (uint32_t)n_embd};
    fwrite(header, 4, 2, out);

    int total_samples = 0;
    int offset = 0;

    while (total_samples < max_samples && offset + batch_size < n_corpus) {
        int bs = std::min(batch_size, n_corpus - offset - 1);
        bs = std::min(bs, max_samples - total_samples);

        // Clear KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        // Create batch
        llama_batch batch = llama_batch_init(bs, 0, 1);
        for (int i = 0; i < bs; i++) {
            batch.token[i] = corpus[offset + i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;  // need embeddings for all positions
        }
        batch.n_tokens = bs;

        // Decode
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "Decode failed at offset %d (ret=%d)\n", offset, ret);
            llama_batch_free(batch);
            offset += bs;
            continue;
        }

        // Extract embeddings for each position
        for (int i = 0; i < bs; i++) {
            float* emb = llama_get_embeddings_ith(ctx, i);
            if (!emb) continue;

            int32_t next_token = corpus[offset + i + 1];  // the token we want to predict

            fwrite(&next_token, 4, 1, out);
            fwrite(emb, sizeof(float), n_embd, out);
            total_samples++;
        }

        llama_batch_free(batch);
        offset += bs;

        if (total_samples % 1000 == 0) {
            printf("  %d/%d samples extracted (offset=%d)\n", total_samples, max_samples, offset);
        }
    }

    // Update header with actual sample count
    fseek(out, 0, SEEK_SET);
    header[0] = (uint32_t)total_samples;
    fwrite(header, 4, 2, out);
    fclose(out);

    printf("Done: %d samples saved to %s (%.1f MB)\n",
           total_samples, output_path,
           (8 + (long)total_samples * (4 + n_embd * 4)) / 1e6);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
