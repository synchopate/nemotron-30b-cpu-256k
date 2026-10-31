/*
 * Extract TOP-K logit values from model for logit-based probe training.
 * No embeddings=true needed — logits are always available.
 *
 * Output: binary file with:
 *   Header: [n_samples:u32, n_features:u32]   (n_features = K hot token logits)
 *   Hot token IDs: [int32 × n_features]
 *   Per sample: [next_token:i32, logit_values[n_features]:float]
 *
 * Usage:
 *   ./rc_probe_extract_logits -m model.gguf -c corpus.bin -o logit_data.bin [-n 50000] [-k 200]
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unordered_map>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* corpus_path = nullptr;
    const char* output_path = "logit_data.bin";
    int max_samples = 50000;
    int n_threads = 6;
    int batch_size = 512;
    int n_hot = 200;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i+1 < argc) corpus_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) max_samples = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) n_hot = atoi(argv[++i]);
    }

    if (!model_path || !corpus_path) {
        fprintf(stderr, "Usage: %s -m model.gguf -c corpus.bin -o output.bin [-n 50000] [-k 200]\n", argv[0]);
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

    // Find top-K hot tokens by frequency
    std::unordered_map<int32_t, int> freq;
    for (int i = 0; i < n_corpus; i++) freq[corpus[i]]++;

    std::vector<std::pair<int,int32_t>> sorted_freq;
    for (auto& [tok, cnt] : freq) sorted_freq.push_back({cnt, tok});
    std::sort(sorted_freq.rbegin(), sorted_freq.rend());

    std::vector<int32_t> hot_tokens;
    for (int i = 0; i < std::min(n_hot, (int)sorted_freq.size()); i++)
        hot_tokens.push_back(sorted_freq[i].second);
    printf("Hot tokens: %d (top by frequency)\n", (int)hot_tokens.size());

    // Load model — NO embeddings needed!
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("Model loaded: vocab=%d\n", nv);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = batch_size + 16;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    // embeddings = false (default) — no overhead!

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // Output file
    int n_features = (int)hot_tokens.size();
    FILE* out = fopen(output_path, "wb");
    if (!out) { fprintf(stderr, "Cannot create output\n"); return 1; }

    // Header
    uint32_t header[2] = {0, (uint32_t)n_features};
    fwrite(header, 4, 2, out);
    // Hot token IDs
    fwrite(hot_tokens.data(), 4, n_features, out);

    int total_samples = 0;
    int offset = 0;

    while (total_samples < max_samples && offset + batch_size < n_corpus) {
        int bs = std::min(batch_size, n_corpus - offset - 1);
        bs = std::min(bs, max_samples - total_samples);

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(bs, 0, 1);
        for (int i = 0; i < bs; i++) {
            batch.token[i] = corpus[offset + i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;  // need logits for all positions
        }
        batch.n_tokens = bs;

        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "Decode failed at offset %d\n", offset);
            llama_batch_free(batch);
            offset += bs;
            continue;
        }

        // Extract logits for hot tokens at each position
        for (int i = 0; i < bs; i++) {
            const float* logits = llama_get_logits_ith(ctx, i);
            if (!logits) continue;

            int32_t next_token = corpus[offset + i + 1];
            fwrite(&next_token, 4, 1, out);

            // Write logit values for each hot token
            for (int h = 0; h < n_features; h++) {
                float val = logits[hot_tokens[h]];
                fwrite(&val, 4, 1, out);
            }
            total_samples++;
        }

        llama_batch_free(batch);
        offset += bs;

        if (total_samples % 5000 == 0)
            printf("  %d/%d samples\n", total_samples, max_samples);
    }

    // Update header
    fseek(out, 0, SEEK_SET);
    header[0] = (uint32_t)total_samples;
    fwrite(header, 4, 2, out);
    fclose(out);

    float size_mb = (8.0f + n_features * 4 + (float)total_samples * (4 + n_features * 4)) / 1e6f;
    printf("Done: %d samples, %d features → %s (%.1f MB)\n",
           total_samples, n_features, output_path, size_mb);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
