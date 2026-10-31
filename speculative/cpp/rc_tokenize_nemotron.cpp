#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

static void tokenize_file(const llama_vocab * vocab, const char * in_path, const char * out_path, std::vector<llama_token> * out_vec = nullptr) {
    std::ifstream ifs(in_path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (text.empty()) return;

    int32_t max_toks = (int32_t)text.size() + 64;
    std::vector<llama_token> toks(max_toks);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(), max_toks, false, true);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), false, true);
    }
    toks.resize(n);

    FILE * f = fopen(out_path, "wb");
    if (f) {
        fwrite(toks.data(), sizeof(int32_t), toks.size(), f);
        fclose(f);
        printf("Tokenized %s -> %s: %d tokens (%.2f MB)\n", in_path, out_path, n, (n * 4.0) / (1024.0 * 1024.0));
    }
    if (out_vec) *out_vec = std::move(toks);
}

int main(int argc, char ** argv) {
    const char * model_path = (argc > 1) ? argv[1] : "./workspace/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf";
    const char * in_txt     = (argc > 2) ? argv[2] : "./workspace/routecrack/bootstrap_corpus.txt";
    const char * out_bin    = (argc > 3) ? argv[3] : "./workspace/routecrack/corpus_nemotron_train.bin";

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model vocab from %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    printf("Loaded Nemotron vocab: %d tokens\n", n_vocab);

    std::vector<llama_token> train_toks;
    tokenize_file(vocab, in_txt, out_bin, &train_toks);
    if (train_toks.empty()) {
        llama_model_free(model);
        return 1;
    }

    std::unordered_map<llama_token, uint32_t> freq;
    for (llama_token t : train_toks) freq[t]++;
    std::vector<std::pair<uint32_t, llama_token>> sorted;
    for (auto & kv : freq) sorted.push_back({kv.second, kv.first});
    std::sort(sorted.rbegin(), sorted.rend());

    printf("\n=== TOP-48 HOT TOKENS FOR NEMOTRON-3.5-30B ===\n");
    uint64_t top48_sum = 0;
    for (size_t i = 0; i < std::min<size_t>(48, sorted.size()); ++i) {
        llama_token tid = sorted[i].second;
        uint32_t cnt = sorted[i].first;
        top48_sum += cnt;
        char buf[128] = {0};
        int l = llama_token_to_piece(vocab, tid, buf, sizeof(buf) - 1, 0, true);
        std::string escaped;
        for (int c = 0; c < l; ++c) {
            unsigned char ch = (unsigned char)buf[c];
            if (ch == '\n') escaped += "\\n";
            else if (ch == '\r') escaped += "\\r";
            else if (ch == '\t') escaped += "\\t";
            else if (ch < 32 || ch >= 127) {
                char hex[8]; snprintf(hex, sizeof(hex), "\\x%02x", ch);
                escaped += hex;
            } else escaped += (char)ch;
        }
        printf("  #%2zu  tok=%6d  freq=%8u (%5.2f%%) -> '%s'\n",
               i + 1, tid, cnt, 100.0 * cnt / train_toks.size(), escaped.c_str());
    }
    printf("Top-48 cumulative share: %.2f%% of all tokens\n", 100.0 * top48_sum / train_toks.size());

    llama_model_free(model);
    return 0;
}
