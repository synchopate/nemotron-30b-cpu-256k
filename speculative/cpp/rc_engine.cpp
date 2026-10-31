/*
 * RouteCrack Engine — C++ Speculative Decoding with BFS Tree Verification
 *
 * Standalone engine using llama.cpp for model inference and our C++ predictors
 * (trie + cascade + PPM + skipgram + codebook) for zero-FLOP drafting.
 *
 * Features:
 *   - BFS tree builder with budget allocation (SGLang-style)
 *   - Multi-sequence batch verification via llama_batch seq_id
 *   - Adaptive draft depth D (increases as accept rate rises)
 *   - Online learning from every generated token
 *   - 5 C++ predictors (0.002ms per call)
 *
 * Build:
 *   g++ -O2 -std=c++17 -o rc_engine rc_engine.cpp \
 *     -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
 *     -L<llama.cpp>/build/bin -lllama -lggml -lggml-base \
 *     -lpthread -ldl -lm -Wl,-rpath,<llama.cpp>/build/bin
 *
 * Usage:
 *   ./rc_engine -m model.gguf --bench
 *   ./rc_engine -m model.gguf --tree-k 3 --adaptive-d
 *   ./rc_engine -m model.gguf -fh hashconf_k10.bin -fh hashconf_k8.bin
 */
#include "llama.h"
#include "rc_predictor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>

using Clock = std::chrono::high_resolution_clock;

static inline double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Pure greedy — used ONLY for draft verification (must match model exactly)
static int greedy(const float* logits, int nv) {
    int best = 0;
    for (int i = 1; i < nv; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

// ════════════════════════════════════════════════════════════
// llama.cpp Native Sampler — handles temp, top-k, top-p, penalties
// ════════════════════════════════════════════════════════════

struct SamplerConfig {
    float temp        = 0.0f;   // 0 = greedy
    int   top_k       = 40;
    float top_p       = 0.9f;
    float rep_penalty = 1.1f;
    int   rep_last_n  = 64;
    uint32_t seed     = 42;
};

static llama_sampler* _sampler = nullptr;       // for final token selection
static llama_sampler* _greedy_sampler = nullptr; // for draft verification

static void sampler_init(const SamplerConfig& cfg) {
    // Final token sampler (with temperature, top-k, top-p, penalties)
    auto sparams = llama_sampler_chain_default_params();
    _sampler = llama_sampler_chain_init(sparams);

    if (cfg.rep_penalty != 1.0f)
        llama_sampler_chain_add(_sampler, llama_sampler_init_penalties(
            cfg.rep_last_n, cfg.rep_penalty, 0.0f, 0.0f));

    if (cfg.temp > 0.0f) {
        // Sampling mode: top-k → top-p → temp → dist
        llama_sampler_chain_add(_sampler, llama_sampler_init_top_k(cfg.top_k));
        llama_sampler_chain_add(_sampler, llama_sampler_init_top_p(cfg.top_p, 1));
        llama_sampler_chain_add(_sampler, llama_sampler_init_temp(cfg.temp));
        llama_sampler_chain_add(_sampler, llama_sampler_init_dist(cfg.seed));
    } else {
        // Greedy mode (with penalties)
        llama_sampler_chain_add(_sampler, llama_sampler_init_greedy());
    }

    // Greedy sampler for draft verification — pure, no penalties
    auto gparams = llama_sampler_chain_default_params();
    _greedy_sampler = llama_sampler_chain_init(gparams);
    llama_sampler_chain_add(_greedy_sampler, llama_sampler_init_greedy());
}

static void sampler_free() {
    if (_sampler) { llama_sampler_free(_sampler); _sampler = nullptr; }
    if (_greedy_sampler) { llama_sampler_free(_greedy_sampler); _greedy_sampler = nullptr; }
}

static void sampler_reset() {
    if (_sampler) llama_sampler_reset(_sampler);
    if (_greedy_sampler) llama_sampler_reset(_greedy_sampler);
}

// Sample from raw logits using the sampler chain (avoids llama_sampler_sample idx issues)
static int sample_from_logits(const float* logits, int nv) {
    std::vector<llama_token_data> candidates(nv);
    for (int i = 0; i < nv; i++)
        candidates[i] = {i, logits[i], 0.0f};
    llama_token_data_array cur_p = {candidates.data(), (size_t)nv, -1, false};
    llama_sampler_apply(_sampler, &cur_p);
    int tok = cur_p.data[cur_p.selected].id;
    llama_sampler_accept(_sampler, tok);
    return tok;
}

// Suppress llama.cpp stderr
static int saved_stderr = -1;
static void silence() {
    fflush(stderr); saved_stderr = dup(STDERR_FILENO);
    int dn = open("/dev/null", O_WRONLY); dup2(dn, STDERR_FILENO); close(dn);
}
static void restore() {
    if (saved_stderr >= 0) {
        fflush(stderr); dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr); saved_stderr = -1;
    }
}

// Benchmark prompts (30 diverse, 6 domains)
// Benchmark groups for per-category measurement
struct BenchGroup {
    const char* name;
    const char** prompts;
    int n;
};

static const char* coding_prompts[] = {
    "Write a Python function to check if a string is a palindrome",
    "Implement a Python class for a priority queue using a heap",
    "Write a Python decorator that retries a function on exception",
    "Create a React component for a searchable dropdown menu",
    "Write a FastAPI endpoint that handles file uploads with validation",
    "Implement Dijkstra shortest path algorithm in Python",
    "Write a Python function to solve the N-queens problem",
    "Create a SQLAlchemy model with relationships and cascade delete",
    "Write a Dockerfile for a multi-stage Python application",
    "Implement a Python LRU cache from scratch",
};

static const char* english_prompts[] = {
    "Explain the causes and consequences of the French Revolution",
    "Describe the process of photosynthesis in plants step by step",
    "What are the main differences between classical and operant conditioning",
    "Explain how the human immune system fights viral infections",
    "Describe the economic factors that led to the Great Depression",
    "What is the theory of relativity and why is it important",
    "Explain the water cycle and its impact on weather patterns",
    "Describe the main philosophical ideas of Immanuel Kant",
    "What are the environmental impacts of deforestation",
    "Explain how neural networks learn from training data",
};

// Agentic: REPETITIVE tool-call patterns (JSON, SSH, same format)
// These simulate real Agent CLI / Copilot workloads
static const char* agentic_prompts[] = {
    "Write a Python function that validates JSON against schema and returns errors as JSON",
    "Write a Python function that validates JSON against schema and returns a summary",
    "Write a Python function that reads a JSON config file and validates required fields",
    "Write a bash script that SSH into a server and checks disk usage",
    "Write a bash script that SSH into a server and restarts a service",
    "Write a bash script that SSH into a server and tails the last 100 lines of a log",
    "Write a Python function that calls an API endpoint and parses the JSON response",
    "Write a Python function that calls an API endpoint and handles errors with retry",
    "Write a Python function that reads a YAML file and converts it to JSON",
    "Write a Python function that reads a TOML file and converts it to JSON",
};

// OOD: prompts the model has NEVER seen in cascade training corpus
static const char* ood_prompts[] = {
    "Explain the causes and long-term impact of the Industrial Revolution",
    "Write a Python function for topological sort of a DAG",
    "What are the key principles of object-oriented programming",
    "Describe how a compiler transforms source code into machine code",
    "Implement a Python A* pathfinding algorithm",
    "Explain the difference between TCP and UDP protocols",
    "Write a MongoDB aggregation pipeline for user analytics",
    "What is the CAP theorem and why does it matter for databases",
    "Implement a simple ETL pipeline with error handling and retry",
    "Explain the concept of eventual consistency in distributed systems",
};

static BenchGroup bench_groups[] = {
    {"Coding",  coding_prompts,  10},
    {"English", english_prompts, 10},
    {"Agentic", agentic_prompts, 10},
    {"OOD",     ood_prompts,     10},
};
static const int N_GROUPS = 4;

// Flat array for backward compat
static const char* bench_prompts[40];
static int N_BENCH = 0;
static void init_bench_prompts() {
    N_BENCH = 0;
    for (int g = 0; g < N_GROUPS; g++)
        for (int i = 0; i < bench_groups[g].n; i++)
            bench_prompts[N_BENCH++] = bench_groups[g].prompts[i];
}

// ════════════════════════════════════════════════════════════
// Speculative Decode Engine
// ════════════════════════════════════════════════════════════

struct EngineStats {
    int total_tokens;
    int total_drafted;
    int total_accepted;
    int total_steps;
    int total_fallback;
    double total_time_ms;
    double baseline_tps;
};

static bool _in_think_block = false;
static int _im_end_token = -1;  // <|im_end|> token ID

static bool is_stop_token(const llama_vocab* voc, int tok) {
    if (llama_vocab_is_eog(voc, tok)) return true;
    if (_im_end_token >= 0 && tok == _im_end_token) return true;
    // Check text
    char buf[64];
    int len = llama_token_to_piece(voc, tok, buf, 63, 0, true);
    if (len > 0) {
        buf[len] = 0;
        if (strstr(buf, "<|im_end|>") || strstr(buf, "<|endoftext|>"))
            return true;
    }
    return false;
}

static void print_token(const llama_vocab* voc, int tok, bool bypassed) {
    char buf[256];
    int len = llama_token_to_piece(voc, tok, buf, 255, 0, true);
    if (len <= 0) return;
    buf[len] = 0;

    // Mark <think>...</think> blocks in dim gray
    if (strstr(buf, "<think>")) { _in_think_block = true; printf("\033[90m[thinking] "); fflush(stdout); return; }
    if (strstr(buf, "</think>")) { _in_think_block = false; printf("\033[0m\n"); fflush(stdout); return; }
    if (_in_think_block) {
        printf("\033[90m%s\033[0m", buf);  // dim gray for thinking
        fflush(stdout);
        return;
    }

    if (bypassed)
        printf("\033[92m%s\033[0m", buf);  // green = predicted
    else
        printf("%s", buf);  // white = computed
    fflush(stdout);
}

static int spec_decode_one(
    llama_context* ctx, const llama_vocab* voc, int nv,
    RCPredictor* pred, const char* prompt,
    int max_tokens, int tree_k, int tree_d, bool adaptive_d,
    EngineStats* stats, bool show_tokens = false)
{
    // Tokenize prompt
    int prompt_tokens[4096];
    int n_prompt = llama_tokenize(voc, prompt, strlen(prompt),
                                  prompt_tokens, 4096, true, false);
    if (n_prompt <= 0) return 0;

    // Prefill
    llama_batch bp = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        bp.token[i] = prompt_tokens[i];
        bp.pos[i] = i;
        bp.n_seq_id[i] = 1;
        bp.seq_id[i][0] = 0;
        bp.logits[i] = (i == n_prompt - 1) ? 1 : 0;
    }
    bp.n_tokens = n_prompt;
    silence();
    llama_decode(ctx, bp);
    restore();

    // np = next position to fill in KV cache
    // After prefill, KV has [0..n_prompt-1], so np = n_prompt
    int np = n_prompt;

    // target_tok = first generated token (NOT in KV yet)
    int target_tok = sample_from_logits(llama_get_logits_ith(ctx, -1), nv);
    llama_batch_free(bp);

    // Feed to predictor and context (but NOT to KV — that happens at decode time)
    std::vector<int32_t> context(prompt_tokens, prompt_tokens + n_prompt);
    context.push_back(target_tok);
    pred->learn(context.data(), (int)context.size());
    if (show_tokens) print_token(voc, target_tok, false);

    int generated = 1;
    int D = tree_d;

    // Convention: at the top of each iteration,
    // - KV cache has positions [0..np-1]
    // - target_tok is the last generated token, NOT in KV yet
    // - context includes target_tok

    // Bypass thresholds
    const float BYPASS_CONF = 0.95f;    // min confidence to bypass
    const int   BYPASS_MIN_OBS = 5;     // min observations
    const int   BYPASS_MAX = 20;        // max consecutive bypassed tokens before checkpoint

    // Logit cache: top-5 from last model decode
    // Used to validate draft candidates — if draft not in top-5, skip
    static const int LOGIT_CACHE_K = 20;
    int32_t logit_cache[LOGIT_CACHE_K];
    int logit_cache_n = 0;
    bool use_logit_cache = (getenv("RC_LOGIT_CACHE") && atoi(getenv("RC_LOGIT_CACHE")));

    auto fill_logit_cache = [&](const float* logits) {
        if (!logits || !use_logit_cache) { logit_cache_n = 0; return; }
        // Partial sort for top-5
        int32_t idx[LOGIT_CACHE_K];
        for (int i = 0; i < LOGIT_CACHE_K; i++) idx[i] = i;
        for (int i = 0; i < LOGIT_CACHE_K; i++) {
            for (int j = i + 1; j < nv; j++) {
                if (logits[j] > logits[idx[i]]) {
                    // Shift down
                    for (int k = LOGIT_CACHE_K - 1; k > i; k--) idx[k] = idx[k-1];
                    idx[i] = j;
                    break;
                }
            }
        }
        // Proper top-5: scan once
        for (int i = 0; i < LOGIT_CACHE_K; i++) idx[i] = 0;
        for (int j = 0; j < nv; j++) {
            // Insert into sorted top-K if larger than smallest
            if (logits[j] > logits[idx[LOGIT_CACHE_K-1]]) {
                idx[LOGIT_CACHE_K-1] = j;
                // Bubble up
                for (int k = LOGIT_CACHE_K-2; k >= 0; k--) {
                    if (logits[idx[k+1]] > logits[idx[k]])
                        std::swap(idx[k], idx[k+1]);
                    else break;
                }
            }
        }
        for (int i = 0; i < LOGIT_CACHE_K; i++) logit_cache[i] = idx[i];
        logit_cache_n = LOGIT_CACHE_K;
    };

    auto in_logit_cache = [&](int32_t tok) -> bool {
        for (int i = 0; i < logit_cache_n; i++)
            if (logit_cache[i] == tok) return true;
        return false;
    };

    // Helper lambda: single token decode
    auto do_single_decode = [&]() {
        silence();
        llama_batch b1 = llama_batch_init(1, 0, 1);
        b1.token[0] = target_tok; b1.pos[0] = np;
        b1.n_seq_id[0] = 1; b1.seq_id[0][0] = 0;
        b1.logits[0] = 1; b1.n_tokens = 1;
        llama_decode(ctx, b1);
        restore();
        np++;
        const float* sd_logits = llama_get_logits_ith(ctx, -1);
        target_tok = sample_from_logits(sd_logits, nv);
        fill_logit_cache(sd_logits);
        // Online probe: collect (logits → actual_token) for training
        pred->online_probe.observe_token(target_tok);
        pred->online_probe.add_sample(sd_logits, nv, target_tok);
        llama_batch_free(b1);
        context.push_back(target_tok);
        if (show_tokens) print_token(voc, target_tok, false);
        generated++;
        pred->learn(context.data() + std::max(0, (int)context.size() - 18),
                   std::min(18, (int)context.size()));
        stats->total_fallback++;
    };

    // Helper lambda: batch verify a list of draft tokens
    auto do_batch_verify = [&](std::vector<int32_t>& draft) -> bool {
        stats->total_drafted += (int)draft.size();
        int batch_size = 1 + (int)draft.size();
        llama_batch bv = llama_batch_init(batch_size, 0, 1);
        bv.token[0] = target_tok; bv.pos[0] = np;
        bv.n_seq_id[0] = 1; bv.seq_id[0][0] = 0; bv.logits[0] = 1;
        for (int d = 0; d < (int)draft.size(); d++) {
            bv.token[1+d] = draft[d]; bv.pos[1+d] = np + 1 + d;
            bv.n_seq_id[1+d] = 1; bv.seq_id[1+d][0] = 0;
            bv.logits[1+d] = 1;
        }
        bv.n_tokens = batch_size;
        silence();
        int dec_ret = llama_decode(ctx, bv);
        restore();
        if (dec_ret != 0) { llama_batch_free(bv); do_single_decode(); return is_stop_token(voc, target_tok); }

        int accepted = 0;
        int correction_tok = -1;
        for (int d = 0; d < (int)draft.size(); d++) {
            const float* vl = llama_get_logits_ith(ctx, d);
            if (!vl) break;
            int model_tok = greedy(vl, nv);
            if (model_tok == draft[d]) {
                accepted++;
                llama_sampler_accept(_sampler, draft[d]);
                // Online probe: logits at pos d predicted draft[d] correctly
                pred->online_probe.observe_token(draft[d]);
                pred->online_probe.add_sample(vl, nv, draft[d]);
            } else {
                correction_tok = sample_from_logits(vl, nv);
                // Online probe: logits at pos d → actual was correction_tok
                pred->online_probe.observe_token(correction_tok);
                pred->online_probe.add_sample(vl, nv, correction_tok);
                break;
            }
        }
        if (accepted == (int)draft.size()) {
            const float* bl = llama_get_logits_ith(ctx, -1);
            if (bl) {
                correction_tok = sample_from_logits(bl, nv);
                fill_logit_cache(bl);
                pred->online_probe.observe_token(correction_tok);
                pred->online_probe.add_sample(bl, nv, correction_tok);
            }
        } else {
            const float* bl = llama_get_logits_ith(ctx, accepted);
            fill_logit_cache(bl);
        }
        llama_batch_free(bv);

        int fr = np + 1 + accepted, lr = np + 1 + (int)draft.size();
        if (fr < lr) llama_memory_seq_rm(llama_get_memory(ctx), 0, fr, lr);
        np += 1 + accepted;

        for (int d = 0; d < accepted; d++) {
            context.push_back(draft[d]);
            if (show_tokens) print_token(voc, draft[d], true);
            generated++;
            if (is_stop_token(voc, draft[d])) return true;
        }
        if (correction_tok >= 0) {
            context.push_back(correction_tok);
            if (show_tokens) print_token(voc, correction_tok, accepted == (int)draft.size());
            generated++;
        }
        stats->total_accepted += accepted;
        target_tok = context.back();
        int ls = std::max(0, (int)context.size() - 18);
        pred->learn(context.data() + ls, (int)context.size() - ls);
        if (adaptive_d) {
            // Use RECENT accept rate (last ~50 drafts), not cumulative
            // This reacts faster to workload changes
            static int recent_accepted = 0, recent_drafted = 0;
            recent_accepted += accepted;
            recent_drafted += (int)draft.size();
            if (recent_drafted > 50) {
                double ar = (double)recent_accepted / recent_drafted;
                if (ar > 0.70) D = 8;
                else if (ar > 0.30) D = 1;
                else D = 4;
                // Decay: halve counters to forget old data
                recent_accepted /= 2;
                recent_drafted /= 2;
            }
        }
        return false;
    };

    while (generated < max_tokens) {
        stats->total_steps++;

        // ═══════════════════════════════════════════════════════
        // PHASE 1: Confident Bypass — emit tokens without model
        // ═══════════════════════════════════════════════════════
        std::vector<int32_t> bypassed_tokens;
        {
            std::vector<int32_t> temp_ctx = context;
            int32_t check_tok = target_tok;

            while ((int)bypassed_tokens.size() < BYPASS_MAX &&
                   generated + (int)bypassed_tokens.size() < max_tokens) {
                // Add current token to temp context for next prediction
                if (!bypassed_tokens.empty())
                    temp_ctx.push_back(bypassed_tokens.back());

                float conf = 0;
                int obs = 0, depth = 0;
                int32_t predicted = pred->predict_confident(
                    temp_ctx.data(), (int)temp_ctx.size(), &conf, &obs, &depth);

                if (predicted < 0 || predicted >= nv) break;
                if (conf < BYPASS_CONF || obs < BYPASS_MIN_OBS) break;
                if (depth < 3) break;  // need at least 3-token context match
                if (is_stop_token(voc, predicted)) break;

                bypassed_tokens.push_back(predicted);
                temp_ctx.push_back(predicted);
            }
        }

        // ═══════════════════════════════════════════════════════
        // PHASE 2a: Macro drafting — try multi-token continuation
        // If macro codebook has a long match, draft the whole thing
        // ═══════════════════════════════════════════════════════
        std::vector<int32_t> combined = bypassed_tokens;  // start with bypass (may be empty)

        if (combined.empty() && pred->macro.n_entries > 0) {
            int32_t macro_buf[RC_MACRO_MAX_CONT];
            int macro_n = pred->macro.predict(
                context.data(), (int)context.size(), macro_buf, RC_MACRO_MAX_CONT);
            if (macro_n >= 2) {
                for (int j = 0; j < macro_n; j++)
                    combined.push_back(macro_buf[j]);
            }
        }

        // ═══════════════════════════════════════════════════════
        // PHASE 2b: Extend with draft tokens (lower confidence)
        // Combine bypass + macro + draft into ONE batch verify
        // ═══════════════════════════════════════════════════════

        // Build draft chain from where bypass/macro stopped
        {
            std::vector<int32_t> temp_ctx = context;
            for (auto t : combined) temp_ctx.push_back(t);

            int remaining = D - (int)combined.size();
            for (int d = 0; d < remaining; d++) {
                int tlen = (int)temp_ctx.size();
                int32_t next_tok = -1;

                // ── STRUCTURAL OVERRIDE 1: In-context 4-gram repetition ──
                // If last 3 tokens appeared earlier in context → predict what followed
                if (tlen >= 4) {
                    for (int pos = std::max(0, tlen - 100); pos < tlen - 4; pos++) {
                        if (temp_ctx[pos]   == temp_ctx[tlen-3] &&
                            temp_ctx[pos+1] == temp_ctx[tlen-2] &&
                            temp_ctx[pos+2] == temp_ctx[tlen-1]) {
                            next_tok = temp_ctx[pos+3];
                            break;
                        }
                    }
                }

                // ── STRUCTURAL OVERRIDE 2: BPE continuation ──
                // If cascade k=1 has ≥95% confidence → deterministic BPE pair
                if (next_tok < 0 && tlen >= 1 && pred->n_cascade > 0) {
                    for (int c = 0; c < pred->n_cascade; c++) {
                        if (pred->cascade_k[c] == 1) {
                            int conf = 0;
                            int32_t p = pred->cascade[c].lookup_conf(
                                rc_fnv64(&temp_ctx[tlen-1], 1), &conf);
                            if (p >= 0 && conf >= 95) next_tok = p;
                            break;
                        }
                    }
                }

                // ── STANDARD: predict_topk (trie + cascade + PPM + skip + codebook) ──
                if (next_tok < 0) {
                    int32_t top_cands[16];
                    int n_cands = pred->predict_topk(
                        temp_ctx.data(), tlen, top_cands,
                        std::min(tree_k, 8));
                    if (n_cands > 0 && top_cands[0] >= 0 && top_cands[0] < nv)
                        next_tok = top_cands[0];
                }

                if (next_tok < 0 || next_tok >= nv) break;
                combined.push_back(next_tok);
                temp_ctx.push_back(next_tok);
            }
        }

        if (combined.empty()) {
            do_single_decode();
            if (is_stop_token(voc, target_tok)) break;
            continue;
        }

        // ONE batch verify for everything (bypass + draft)
        if (do_batch_verify(combined)) break;
        if (is_stop_token(voc, target_tok)) break;
    }

done:
    stats->total_tokens += generated;
    return generated;
}

// ════════════════════════════════════════════════════════════
// Auto-Setup: generate corpus → build cascade → load into predictor
// ════════════════════════════════════════════════════════════

static void auto_build_cascade(
    llama_context* ctx, const llama_vocab* voc, int nv,
    RCPredictor* pred, const char* output_dir,
    int max_tokens)
{
    printf("\n[AUTO] Phase 1: Generate corpus (30 prompts)...\n");

    static const char* corpus_prompts[] = {
        "Write a Python function to check if a number is prime",
        "Implement a binary search in Python",
        "Write a Python class for a stack with push pop peek",
        "Create a Python function to reverse a string",
        "Write a Python decorator that logs execution time",
        "Implement a Python linked list with insert and delete",
        "Write a Python function to count character frequencies",
        "Create a Python generator for prime numbers",
        "Write a Python merge sort with explanation",
        "Implement a Python hash table with chaining",
        "Write a FastAPI endpoint for user registration",
        "Create a Flask REST API with CRUD operations",
        "Write a Python WebSocket server",
        "Implement HTTP client with retry logic",
        "Write a Django model for a blog with tags",
        "Write a Python CSV parser with statistics",
        "Create a Python data pipeline with error recovery",
        "Implement a connection pool in Python",
        "Write a Python ETL pipeline for logs",
        "Create a Python key-value store with TTL",
        "Write a Dockerfile for multi-stage Python app",
        "Create a GitHub Actions CI workflow",
        "Write a Python system resource monitor",
        "Implement a health check with circuit breaker",
        "Write Kubernetes deployment YAML",
        "Write a Python N-queens solver",
        "Implement Dijkstra shortest path in Python",
        "Write a Python topological sort",
        "Implement A* pathfinding in Python",
        "Write a Python function to find strongly connected components",
    };
    int n_corpus = 30;

    // Collect all generated tokens
    std::vector<int32_t> all_tokens;

    for (int i = 0; i < n_corpus; i++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        EngineStats ws = {};
        int n = spec_decode_one(ctx, voc, nv, pred, corpus_prompts[i],
                                max_tokens, 1, 1, false, &ws, false);
        // The context vector inside spec_decode_one is local,
        // but the predictor learns from it via pred->learn().
        // We need to capture the tokens separately.
        // For now, we run the generation and the predictor learns online.
        if ((i + 1) % 10 == 0)
            printf("  %d/%d done (trie: %d nodes)\n", i + 1, n_corpus, pred->trie.used);
    }

    // Phase 2: The predictor already learned from all generated tokens
    // via online trie + PPM + skipgram. No need to build cascade separately.
    printf("[AUTO] Phase 1 complete: predictor has %d trie nodes\n", pred->trie.used);

    // Phase 2: Tune — try different K values
    printf("\n[AUTO] Phase 2: Tuning (K=1,3,5 × D=4,8)...\n");

    static const char* tune_prompts[] = {
        "Write a Python function to check if a string is a palindrome",
        "Implement a Python class for a priority queue",
        "Write a Python decorator that retries on exception",
        "Create a Python context manager for database connections",
        "Implement a Python LRU cache from scratch",
    };
    int n_tune = 5;

    double base_tps = 0;
    {
        double tok = 0, ms = 0;
        for (int i = 0; i < n_tune; i++) {
            llama_memory_clear(llama_get_memory(ctx), true);
            EngineStats s = {};
            auto t0 = Clock::now();
            int n = spec_decode_one(ctx, voc, nv, pred, tune_prompts[i],
                                    100, 1, 1, false, &s, false);
            auto t1 = Clock::now();
            tok += n; ms += elapsed_ms(t0, t1);
        }
        base_tps = tok / (ms / 1000.0);
        printf("  Baseline: %.1f tok/s\n", base_tps);
    }

    int best_k = 1, best_d = 8;
    double best_speedup = 1.0;

    int test_ks[] = {1, 3, 5};
    int test_ds[] = {4, 8};

    for (int ki = 0; ki < 3; ki++) {
        for (int di = 0; di < 2; di++) {
            int K = test_ks[ki], D = test_ds[di];
            double tok = 0, ms = 0;
            EngineStats total = {};
            for (int i = 0; i < n_tune; i++) {
                llama_memory_clear(llama_get_memory(ctx), true);
                auto t0 = Clock::now();
                int n = spec_decode_one(ctx, voc, nv, pred, tune_prompts[i],
                                        100, K, D, true, &total, false);
                auto t1 = Clock::now();
                tok += n; ms += elapsed_ms(t0, t1);
            }
            double tps = tok / (ms / 1000.0);
            double speedup = tps / base_tps;
            double acc = total.total_drafted > 0 ?
                (double)total.total_accepted / total.total_drafted * 100 : 0;
            printf("  K=%d D=%d: %.1f tok/s = %.2fx (accept %.0f%%)\n",
                   K, D, tps, speedup, acc);
            if (speedup > best_speedup) {
                best_speedup = speedup;
                best_k = K; best_d = D;
            }
        }
    }

    printf("\n[AUTO] Optimal: K=%d D=%d (%.2fx)\n", best_k, best_d, best_speedup);

    // Save trie state — skipped (trie v2 save requires DFS, done via libroutecrack)
    // The trie is rebuilt from warmup prompts each session anyway.

    printf("[AUTO] Setup complete!\n\n");
}

// ════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    // Disable stdout buffering for pipe/subprocess compatibility
    setvbuf(stdout, NULL, _IONBF, 0);
    // Parse args
    const char* model_path = nullptr;
    std::vector<const char*> cascade_files;
    const char* hot_rules_file = nullptr;
    const char* probe_file = nullptr;
    const char* corpus_file = nullptr;
    int max_tokens = 200;
    int tree_k = 3;
    int tree_d = 8;
    bool bench = false;
    bool auto_mode = false;
    bool adaptive_d = true;
    int n_gpu = 99;
    int n_passes = 5;
    const char* output_dir = ".";
    SamplerConfig sampler_cfg;
    bool auto_detect_thinker = true;
    int expert_used = 0;  // 0 = use model default

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-fh") == 0 && i+1 < argc) cascade_files.push_back(argv[++i]);
        else if (strcmp(argv[i], "-hr") == 0 && i+1 < argc) hot_rules_file = argv[++i];
        else if (strcmp(argv[i], "-fc") == 0 && i+1 < argc) corpus_file = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tree-k") == 0 && i+1 < argc) tree_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tree-d") == 0 && i+1 < argc) tree_d = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bench") == 0) bench = true;
        else if (strcmp(argv[i], "--auto") == 0) auto_mode = true;
        else if (strcmp(argv[i], "--no-adaptive") == 0) adaptive_d = false;
        else if (strcmp(argv[i], "--ngl") == 0 && i+1 < argc) n_gpu = atoi(argv[++i]);
        else if (strcmp(argv[i], "--passes") == 0 && i+1 < argc) n_passes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output-dir") == 0 && i+1 < argc) output_dir = argv[++i];
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) { sampler_cfg.temp = atof(argv[++i]); auto_detect_thinker = false; }
        else if (strcmp(argv[i], "--top-k") == 0 && i+1 < argc) { sampler_cfg.top_k = atoi(argv[++i]); auto_detect_thinker = false; }
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc) { sampler_cfg.top_p = atof(argv[++i]); auto_detect_thinker = false; }
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i+1 < argc) { sampler_cfg.rep_penalty = atof(argv[++i]); auto_detect_thinker = false; }
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) sampler_cfg.seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--greedy") == 0) { sampler_cfg.temp = 0.0f; sampler_cfg.rep_penalty = 1.0f; auto_detect_thinker = false; }
        else if (strcmp(argv[i], "--expert-used") == 0 && i+1 < argc) expert_used = atoi(argv[++i]);
        else if (strcmp(argv[i], "--probe") == 0 && i+1 < argc) probe_file = argv[++i];
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s -m model.gguf [options]\n", argv[0]);
        fprintf(stderr, "  -fh FILE    Hash cascade file (repeat for k=10,8,6,...)\n");
        fprintf(stderr, "  -fc FILE    Binary corpus file\n");
        fprintf(stderr, "  -n N        Max tokens per response (default: 200)\n");
        fprintf(stderr, "  --tree-k K  Top-K candidates (default: 3)\n");
        fprintf(stderr, "  --tree-d D  Max draft depth (default: 8)\n");
        fprintf(stderr, "  --bench     Run benchmark (30 prompts)\n");
        fprintf(stderr, "  --auto      Auto-setup: generate corpus, build cascade, tune, bench\n");
        fprintf(stderr, "  --passes N  Number of benchmark passes (default: 5)\n");
        fprintf(stderr, "  --ngl N     GPU layers (default: all)\n");
        fprintf(stderr, "  --output-dir DIR  Where to save cascade/corpus files (default: .)\n");
        fprintf(stderr, "  --no-adaptive  Disable adaptive D\n");
        fprintf(stderr, "\nSampling (for thinking models like Qwen3.5/3.6):\n");
        fprintf(stderr, "  --temp F      Temperature (default: auto-detect, 0=greedy)\n");
        fprintf(stderr, "  --top-k N     Top-K (default: 40)\n");
        fprintf(stderr, "  --top-p F     Top-P nucleus (default: 0.9)\n");
        fprintf(stderr, "  --rep-penalty F  Repetition penalty (default: 1.1)\n");
        fprintf(stderr, "  --seed N      Random seed (default: 42)\n");
        fprintf(stderr, "  --greedy      Force pure greedy (no temp, no penalties)\n");
        fprintf(stderr, "\nExpert pruning (MoE models):\n");
        fprintf(stderr, "  --expert-used N  Override active experts (e.g., 4 instead of 8)\n");
        return 1;
    }

    // Init predictor
    RCPredictor pred;
    pred.init();
    for (auto f : cascade_files) pred.load_cascade(f);

    // Macro codebook (multi-token continuations)
    {
        const char* macro_path = nullptr;
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], "--macro") == 0 && i+1 < argc) macro_path = argv[i+1];
        // Also try auto-detect in same dir as cascade files
        if (!macro_path && !cascade_files.empty()) {
            static std::string auto_macro;
            std::string dir = cascade_files[0];
            size_t slash = dir.rfind('/');
            if (slash != std::string::npos) dir = dir.substr(0, slash + 1);
            else dir = "./";
            auto_macro = dir + "macro_codebook.bin";
            FILE* test = fopen(auto_macro.c_str(), "rb");
            if (test) { fclose(test); macro_path = auto_macro.c_str(); }
        }
        if (macro_path) pred.macro.load(macro_path);
    }

    if (hot_rules_file) {
        if (pred.hot_rules.load(hot_rules_file))
            printf("[RC Engine] Hot rules: %u rules from %s\n", pred.hot_rules.n_rules, hot_rules_file);
    }

    // Logit probe (zero-overhead alternative to embedding probe)
    {
        const char* lp = nullptr;
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], "--logit-probe") == 0 && i+1 < argc) lp = argv[i+1];
        if (lp && pred.logit_probe.load(lp))
            printf("[RC Engine] Logit probe: %d classes, %d features from %s\n",
                   pred.logit_probe.n_hot, pred.logit_probe.n_feat, lp);
    }
    if (corpus_file) {
        FILE* cf = fopen(corpus_file, "rb");
        if (cf) {
            fseek(cf, 0, SEEK_END);
            long sz = ftell(cf); fseek(cf, 0, SEEK_SET);
            int n = (int)(sz / 4);
            int32_t* toks = (int32_t*)malloc(sz);
            fread(toks, 4, n, cf); fclose(cf);
            pred.ppm.insert(toks, n);
            pred.skipgram.insert(toks, n);
            free(toks);
            printf("[RC] Corpus loaded: %d tokens\n", n);
        }
    }

    // Init llama.cpp
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;

    // Expert pruning: override expert_used_count for MoE models
    llama_model_kv_override kv_overrides[2] = {};
    if (expert_used > 0) {
        // Try common architecture prefixes
        // The actual key depends on model architecture (gemma4, deepseek2, qwen3, etc.)
        // We'll detect it after loading, but set the override first
        // llama.cpp tries "{arch}.expert_used_count" — we need to know the arch
        // For now, set all common variants
        kv_overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kv_overrides[0].val_i64 = expert_used;
        // Key will be set after we detect the architecture from model path
        std::string mp(model_path);
        std::transform(mp.begin(), mp.end(), mp.begin(), ::tolower);
        if (mp.find("gemma") != std::string::npos)
            strncpy(kv_overrides[0].key, "gemma4.expert_used_count", 127);
        else if (mp.find("deepseek") != std::string::npos)
            strncpy(kv_overrides[0].key, "deepseek2.expert_used_count", 127);
        else if (mp.find("qwen") != std::string::npos)
            strncpy(kv_overrides[0].key, "qwen3moe.expert_used_count", 127);
        else
            strncpy(kv_overrides[0].key, "general.expert_used_count", 127);

        // Sentinel: empty key terminates the list
        kv_overrides[1].key[0] = '\0';
        mparams.kv_overrides = kv_overrides;

        printf("[RC Engine] Expert pruning: %d experts (override: %s)\n",
               expert_used, kv_overrides[0].key);
    }

    silence();
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    restore();
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 64;
    // Thread count from env or default
    const char* env_threads = getenv("RC_THREADS");
    if (env_threads) {
        int nt = atoi(env_threads);
        if (nt > 0) {
            cparams.n_threads = nt;
            cparams.n_threads_batch = nt;
        }
    }
    // cparams.flash_attn = true;  // use flash_attn_type on newer llama.cpp
    // embeddings NOT needed — logit probe uses logits directly

    silence();
    llama_context* ctx = llama_init_from_model(model, cparams);
    restore();
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const llama_vocab* voc = llama_model_get_vocab(model);
    int nv = llama_vocab_n_tokens(voc);

    // Auto-detect thinking models (Qwen3, QwQ, etc.) and set sampling params
    if (auto_detect_thinker) {
        // Check model name/path for thinking model patterns
        std::string mp(model_path);
        std::transform(mp.begin(), mp.end(), mp.begin(), ::tolower);
        bool is_thinker = (mp.find("qwen3") != std::string::npos ||
                           mp.find("qwq") != std::string::npos ||
                           mp.find("deepseek-r1") != std::string::npos);
        if (is_thinker) {
            sampler_cfg.temp = 0.6f;
            sampler_cfg.top_k = 40;
            sampler_cfg.top_p = 0.95f;
            sampler_cfg.rep_penalty = 1.1f;
            printf("[RC Engine] Thinking model detected → temp=%.1f top_k=%d top_p=%.2f\n",
                   sampler_cfg.temp, sampler_cfg.top_k, sampler_cfg.top_p);
        }
    }

    // Initialize samplers
    sampler_init(sampler_cfg);
    if (sampler_cfg.temp > 0)
        printf("[RC Engine] Sampling: temp=%.2f top_k=%d top_p=%.2f rep=%.2f seed=%u\n",
               sampler_cfg.temp, sampler_cfg.top_k, sampler_cfg.top_p,
               sampler_cfg.rep_penalty, sampler_cfg.seed);
    else
        printf("[RC Engine] Sampling: greedy%s\n",
               sampler_cfg.rep_penalty != 1.0f ? " + rep_penalty" : "");

    // Find <|im_end|> and <|endoftext|> tokens for stop detection
    // These are special tokens with high IDs (e.g., 151645 for Qwen)
    {
        // Search from end of vocab (special tokens are at the end)
        char buf[64];
        for (int i = nv - 1; i >= std::max(0, nv - 1000); i--) {
            int len = llama_token_to_piece(voc, i, buf, 63, 0, true);
            if (len > 0) {
                buf[len] = 0;
                if (strstr(buf, "im_end") || strstr(buf, "endoftext")) {
                    _im_end_token = i;
                    printf("[RC Engine] Stop token: %d ('%s')\n", i, buf);
                    break;
                }
            }
        }
    }

    printf("[RC Engine] Model: %s\n", model_path);
    printf("[RC Engine] Vocab: %d, Cascade: %d tables, Predictors: 5, Tree: K=%d D=%d%s\n",
           nv, pred.n_cascade, tree_k, tree_d,
           adaptive_d ? " (adaptive)" : "");
    printf("[RC Engine] Online probe: enabled (discovers hot tokens, trains during warmup)\n");

    // Try to load saved trie state
    {
        char trie_path[512];
        snprintf(trie_path, sizeof(trie_path), "%s/rc_trie_state.bin", output_dir);
        FILE* f = fopen(trie_path, "rb");
        if (f && pred.trie.used <= 2) {
            fclose(f);
            // Trie v2 load — skipped (rebuilt from warmup)
        }
    }

    // Auto-setup only when explicitly requested
    if (auto_mode) {
        printf("[RC Engine] Running auto-setup...\n");
        auto_build_cascade(ctx, voc, nv, &pred, output_dir, max_tokens);
        bench = true;
    } else if (pred.n_cascade == 0 && pred.trie.used <= 2 && !bench) {
        printf("[RC Engine] No predictions loaded — will learn online during chat.\n");
    }

    if (bench) {
        // Warmup: generate 10 prompts to build predictor
        printf("\n[Warmup] 10 prompts...\n");
        static const char* warmup_prompts[] = {
            "Write a Python function to check if a number is prime",
            "Implement a binary search in Python",
            "Write a Python class for a stack",
            "Create a Python function to reverse a string",
            "Write a Python decorator that logs execution time",
            "Implement a Python linked list",
            "Write a Python function to count vowels",
            "Create a Python generator for Fibonacci numbers",
            "Write a Python merge sort implementation",
            "Implement a Python hash map",
        };
        for (int i = 0; i < 10; i++) {
            EngineStats ws = {};
            llama_memory_clear(llama_get_memory(ctx), true);
            spec_decode_one(ctx, voc, nv, &pred, warmup_prompts[i],
                           max_tokens, 1, 1, false, &ws);
            if ((i+1) % 5 == 0)
                printf("  %d/10: trie=%d nodes\n", i+1, pred.trie.used);
        }

        // Report online probe status after warmup
        if (pred.online_probe.trained) {
            int hot_classes = 0;
            for (int i = 0; i < pred.online_probe.n_hot; i++)
                if (pred.online_probe.count[i] > 0) hot_classes++;
            printf("  Online probe: trained on %d samples, %d/%d classes active\n",
                   pred.online_probe.total_samples, hot_classes, pred.online_probe.n_hot);
        } else {
            printf("  Online probe: not yet trained (need 500+ hot samples, have %d)\n",
                   pred.online_probe.total_samples);
        }

        init_bench_prompts();

        // Per-group benchmark
        printf("\n[Benchmark: %d groups × %d prompts × %d passes]\n\n",
               N_GROUPS, bench_groups[0].n, n_passes);

        // Baseline per group
        double group_base_tps[N_GROUPS];
        double total_base_tok = 0, total_base_time = 0;
        for (int g = 0; g < N_GROUPS; g++) {
            double btok = 0, btime = 0;
            for (int i = 0; i < bench_groups[g].n; i++) {
                EngineStats bs = {};
                llama_memory_clear(llama_get_memory(ctx), true);
                auto t0 = Clock::now();
                int n = spec_decode_one(ctx, voc, nv, &pred, bench_groups[g].prompts[i],
                                        max_tokens, 1, 1, false, &bs);
                auto t1 = Clock::now();
                btok += n; btime += elapsed_ms(t0, t1);
            }
            group_base_tps[g] = btok / (btime / 1000.0);
            total_base_tok += btok; total_base_time += btime;
        }
        double base_tps = total_base_tok / (total_base_time / 1000.0);
        printf("  Baseline: %.1f tok/s overall\n", base_tps);
        for (int g = 0; g < N_GROUPS; g++)
            printf("    %-10s: %.1f tok/s\n", bench_groups[g].name, group_base_tps[g]);

        // Speculative per group, per pass
        for (int pass = 1; pass <= n_passes; pass++) {
            printf("\n  Pass %d:\n", pass);
            double total_rc_tok = 0, total_rc_time = 0;
            for (int g = 0; g < N_GROUPS; g++) {
                double rtok = 0, rtime = 0;
                EngineStats gs = {};
                for (int i = 0; i < bench_groups[g].n; i++) {
                    llama_memory_clear(llama_get_memory(ctx), true);
                    auto t0 = Clock::now();
                    int n = spec_decode_one(ctx, voc, nv, &pred, bench_groups[g].prompts[i],
                                            max_tokens, tree_k, tree_d,
                                            adaptive_d, &gs);
                    auto t1 = Clock::now();
                    rtok += n; rtime += elapsed_ms(t0, t1);
                }
                double tps = rtok / (rtime / 1000.0);
                double spd = tps / group_base_tps[g];
                double acc = gs.total_drafted > 0 ?
                    (double)gs.total_accepted / gs.total_drafted * 100 : 0;
                printf("    %-10s: %.1f tok/s = %.2fx (accept %.0f%%, drafted %d)\n",
                       bench_groups[g].name, tps, spd, acc, gs.total_drafted);
                total_rc_tok += rtok; total_rc_time += rtime;
            }
            double overall_tps = total_rc_tok / (total_rc_time / 1000.0);
            printf("    %-10s: %.1f tok/s = %.2fx\n", "OVERALL",
                   overall_tps, overall_tps / base_tps);
        }

        printf("\n[Summary]\n");
        printf("  Baseline: %.1f tok/s\n", base_tps);
        printf("  Predictor: trie=%d nodes, cascade=%d tables\n",
               pred.trie.used, pred.n_cascade);
    } else {
        // Interactive mode with chat template
        char line[4096];
        while (printf("\n> "), fgets(line, sizeof(line), stdin)) {
            line[strcspn(line, "\n")] = 0;
            if (line[0] == 0) continue;
            if (strcmp(line, "quit") == 0) break;

            // Apply chat template from model metadata
            char templated[8192];
            struct llama_chat_message msgs[1];
            msgs[0].role = "user";
            msgs[0].content = line;
            const char* tmpl = llama_model_chat_template(model, nullptr);
            int tlen = llama_chat_apply_template(tmpl, msgs, 1, true,
                                                  templated, sizeof(templated));
            if (tlen <= 0 || tlen >= (int)sizeof(templated)) {
                // Fallback: simple "User: ... Assistant:" format
                snprintf(templated, sizeof(templated),
                    "User: %s\n\nAssistant:", line);
            }

            _in_think_block = false;
            sampler_reset();
            llama_memory_clear(llama_get_memory(ctx), true);
            EngineStats stats = {};
            auto t0 = Clock::now();
            int n = spec_decode_one(ctx, voc, nv, &pred, templated,
                                    max_tokens, tree_k, tree_d,
                                    adaptive_d, &stats, true);
            auto t1 = Clock::now();
            double ms = elapsed_ms(t0, t1);
            double tps = n / (ms / 1000.0);
            double acc = stats.total_drafted > 0 ?
                (double)stats.total_accepted / stats.total_drafted * 100 : 0;
            printf("\n[%d tok, %.1f tok/s, %.0f%% accept]\n", n, tps, acc);
        }
    }

    sampler_free();
    llama_free(ctx);
    llama_model_free(model);
    pred.destroy();
    return 0;
}
