/*
 * main.c — LLM-MoE Runtime Entry Point
 *
 * Usage:
 *   ./llm-moe <model.bin> "prompt text"
 *   ./llm-moe <model.bin> --interactive
 *
 * Features:
 *   - Text generation with temperature, top-k, top-p sampling
 *   - Interactive chat mode
 *   - Structured reasoning mode
 *   - Performance timing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "model.h"
#include "tokenizer.h"
#include "reasoning.h"
#include "math_ops.h"

/* ═══════════════════════════════════════════════════════════════════════
 * Sampling
 * ═══════════════════════════════════════════════════════════════════════*/

/* Simple RNG (xorshift64) for portability */
static uint64_t rng_state = 0;

static void rng_seed(uint64_t seed)
{
    rng_state = seed;
    if (rng_state == 0) rng_state = 1;
}

static uint64_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static float rng_float(void)
{
    return (float)(rng_next() >> 11) / (float)(1ULL << 53);
}

/* ── Temperature + Top-K + Top-P Sampling ──────────────────────────── */

/* Sort helper for top-k/top-p */
typedef struct {
    float val;
    int   idx;
} TokenProb;

static int cmp_token_prob_desc(const void *a, const void *b)
{
    float va = ((const TokenProb *)a)->val;
    float vb = ((const TokenProb *)b)->val;
    if (vb > va) return 1;
    if (vb < va) return -1;
    return 0;
}

static int sample_token(float *logits, int vocab_size,
                        float temperature, int top_k, float top_p)
{
    /* Greedy */
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    /* Apply temperature */
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }

    /* Softmax */
    softmax(logits, vocab_size);

    /* Top-K: keep only top_k highest probability tokens */
    if (top_k > 0 && top_k < vocab_size) {
        /* Find the k-th largest value */
        /* Sort indices by probability (descending) */
        TokenProb *tp = (TokenProb *)malloc((size_t)vocab_size * sizeof(TokenProb));
        if (!tp) return 0;

        for (int i = 0; i < vocab_size; i++) {
            tp[i].val = logits[i];
            tp[i].idx = i;
        }
        qsort(tp, (size_t)vocab_size, sizeof(TokenProb), cmp_token_prob_desc);

        /* Zero out everything below top_k */
        for (int i = top_k; i < vocab_size; i++) {
            logits[tp[i].idx] = 0.0f;
        }

        /* Re-normalize */
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += logits[i];
        if (sum > 0.0f) {
            for (int i = 0; i < vocab_size; i++) logits[i] /= sum;
        }

        /* Top-P (nucleus) within top-k */
        if (top_p < 1.0f) {
            float cumsum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                cumsum += tp[i].val;
                if (cumsum >= top_p) {
                    /* Zero everything after this point */
                    for (int j = i + 1; j < top_k; j++) {
                        logits[tp[j].idx] = 0.0f;
                    }
                    break;
                }
            }

            /* Re-normalize again */
            sum = 0.0f;
            for (int i = 0; i < vocab_size; i++) sum += logits[i];
            if (sum > 0.0f) {
                for (int i = 0; i < vocab_size; i++) logits[i] /= sum;
            }
        }

        free(tp);
    }

    /* Sample from the distribution */
    float r = rng_float();
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }

    return vocab_size - 1;  /* fallback */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Tokenizer Loading (from model binary)
 * ═══════════════════════════════════════════════════════════════════════*/

static int load_tokenizer_from_model(Tokenizer *tok, const char *filepath)
{
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;

    /* Skip: magic (4) + version (4) + config (15 fields × 4 bytes = 60) */
    /* Config has 11 uint32 + 2 float + 4 uint32 = 17 * 4 = 68 total */
    /* Actually let's count: vocab_size..eos_id = 9 uint32 + 2 floats + 4 uint32 = 15 fields */
    fseek(fp, 4 + 4, SEEK_SET);  /* magic + version */

    /* Skip 11 uint32_t + 2 float + 4 uint32_t = 15 × 4 = 60 bytes */
    fseek(fp, 60, SEEK_CUR);  /* config section */

    /* Read tokenizer header */
    uint32_t vocab_size, n_merges;
    if (fread(&vocab_size, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&n_merges, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return -1; }

    int ret = tokenizer_load_from_fp(tok, fp, (int)vocab_size, (int)n_merges);
    fclose(fp);
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Text Generation
 * ═══════════════════════════════════════════════════════════════════════*/

static void generate(Model *model, RunState *state, Tokenizer *tok,
                     const char *prompt, int max_tokens,
                     float temperature, int top_k, float top_p,
                     int use_reasoning)
{
    Config *cfg = &model->config;
    int vocab_size = (int)cfg->vocab_size;

    /* Encode prompt */
    int prompt_tokens[4096];
    int n_prompt = tokenizer_encode(tok, prompt, prompt_tokens, 4096, 1,
                                    (int)cfg->bos_id, (int)cfg->eos_id);

    if (n_prompt <= 0) {
        fprintf(stderr, "Error: empty prompt after tokenization\n");
        return;
    }

    /* Remove trailing EOS from prompt (we want to continue generating) */
    if (prompt_tokens[n_prompt - 1] == (int)cfg->eos_id) {
        n_prompt--;
    }

    fprintf(stderr, "Prompt: %d tokens\n", n_prompt);
    fprintf(stderr, "Generating up to %d tokens...\n\n", max_tokens);

    /* Reasoning state */
    ReasoningState rs;
    if (use_reasoning) {
        reasoning_init(&rs, tok);
    }

    /* Reset KV caches */
    for (uint32_t l = 0; l < cfg->n_layers; l++) {
        state->kv_caches[l].length = 0;
    }

    /* Timing */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int total_tokens = 0;

    /* Prefill: process all prompt tokens */
    for (int i = 0; i < n_prompt; i++) {
        forward(model, state, prompt_tokens[i], i);

        /* Print prompt text */
        const char *s = tokenizer_decode(tok, prompt_tokens[i]);
        if (strcmp(s, "<bos>") != 0) {
            printf("%s", s);
        }
        total_tokens++;
    }
    fflush(stdout);

    /* Generate */
    int pos = n_prompt;
    int prev_token = prompt_tokens[n_prompt - 1];

    for (int i = 0; i < max_tokens; i++) {
        /* Forward pass for next token prediction */
        float *logits = forward(model, state, prev_token, pos);

        /* Sample next token */
        int next_token = sample_token(logits, vocab_size,
                                      temperature, top_k, top_p);

        /* Check EOS */
        if (next_token == (int)cfg->eos_id) {
            break;
        }

        /* Structured reasoning */
        if (use_reasoning) {
            int r = reasoning_process_token(&rs, next_token, logits, vocab_size);
            if (r == -1) {
                /* Reasoning complete */
                reasoning_print_summary(&rs, tok);
            }
        }

        /* Print token */
        const char *s = tokenizer_decode(tok, next_token);

        /* Handle special tokens display */
        if (next_token < 9) {  /* special token range */
            fprintf(stderr, "[%s]", s);
        } else {
            /* Decode byte tokens: <0xHH> → actual byte */
            if (s[0] == '<' && s[1] == '0' && s[2] == 'x') {
                int byte_val = 0;
                if (sscanf(s, "<0x%02X>", &byte_val) == 1) {
                    printf("%c", (char)byte_val);
                } else {
                    printf("%s", s);
                }
            } else {
                printf("%s", s);
            }
        }
        fflush(stdout);

        prev_token = next_token;
        pos++;
        total_tokens++;
    }

    printf("\n");
    fflush(stdout);

    /* Timing */
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (double)(t_end.tv_sec - t_start.tv_sec) +
                     (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    int gen_tokens = total_tokens - n_prompt;

    fprintf(stderr, "\n--- Generation Stats ---\n");
    fprintf(stderr, "Prompt tokens:    %d\n", n_prompt);
    fprintf(stderr, "Generated tokens: %d\n", gen_tokens);
    fprintf(stderr, "Total time:       %.2f s\n", elapsed);
    if (elapsed > 0) {
        fprintf(stderr, "Throughput:       %.1f tok/s\n",
                (double)gen_tokens / elapsed);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Interactive Mode
 * ═══════════════════════════════════════════════════════════════════════*/

static void interactive_mode(Model *model, RunState *state, Tokenizer *tok,
                             float temperature, int top_k, float top_p)
{
    char input[4096];

    printf("LLM-MoE Interactive Mode\n");
    printf("Type your prompt and press Enter. Type 'quit' to exit.\n");
    printf("Commands: /reasoning (toggle structured reasoning)\n\n");

    int use_reasoning = 0;

    while (1) {
        printf("> ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;

        /* Remove trailing newline */
        size_t len = strlen(input);
        if (len > 0 && input[len - 1] == '\n') input[len - 1] = '\0';

        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;

        if (strcmp(input, "/reasoning") == 0) {
            use_reasoning = !use_reasoning;
            printf("Structured reasoning: %s\n", use_reasoning ? "ON" : "OFF");
            continue;
        }

        if (strlen(input) == 0) continue;

        /* Reset KV caches for each new prompt */
        for (uint32_t l = 0; l < model->config.n_layers; l++) {
            state->kv_caches[l].length = 0;
            memset(state->kv_caches[l].key_cache, 0,
                   (size_t)model->config.max_seq_len *
                   model->config.n_heads * model->config.head_dim * sizeof(float));
            memset(state->kv_caches[l].value_cache, 0,
                   (size_t)model->config.max_seq_len *
                   model->config.n_heads * model->config.head_dim * sizeof(float));
        }

        printf("\n");
        generate(model, state, tok, input, 256,
                 temperature, top_k, top_p, use_reasoning);
        printf("\n");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Usage
 * ═══════════════════════════════════════════════════════════════════════*/

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <model.bin> [options] [prompt]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --interactive, -i    Interactive chat mode\n");
    fprintf(stderr, "  --reasoning, -r      Enable structured reasoning\n");
    fprintf(stderr, "  --temperature <f>    Sampling temperature (default: 0.8)\n");
    fprintf(stderr, "  --top-k <n>          Top-K filtering (default: 50)\n");
    fprintf(stderr, "  --top-p <f>          Top-P nucleus sampling (default: 0.9)\n");
    fprintf(stderr, "  --max-tokens <n>     Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  --seed <n>           Random seed (default: time-based)\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s model.bin \"Hello, world!\"\n", prog);
    fprintf(stderr, "  %s model.bin -i\n", prog);
    fprintf(stderr, "  %s model.bin --temperature 0.5 --top-k 40 \"Explain AI\"\n", prog);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Parse arguments */
    const char *model_path = argv[1];
    const char *prompt = NULL;
    int interactive = 0;
    int use_reasoning = 0;
    float temperature = 0.8f;
    int top_k = 50;
    float top_p = 0.9f;
    int max_tokens = 256;
    uint64_t seed = (uint64_t)time(NULL);

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--interactive") == 0 || strcmp(argv[i], "-i") == 0) {
            interactive = 1;
        } else if (strcmp(argv[i], "--reasoning") == 0 || strcmp(argv[i], "-r") == 0) {
            use_reasoning = 1;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (argv[i][0] != '-') {
            prompt = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Initialize RNG */
    rng_seed(seed);

    fprintf(stderr, "╔══════════════════════════════════════╗\n");
    fprintf(stderr, "║       LLM-MoE Runtime (C99)         ║\n");
    fprintf(stderr, "╚══════════════════════════════════════╝\n\n");

    /* ── Load Model ── */
    fprintf(stderr, "Loading model: %s\n", model_path);
    Model model = {0};
    if (model_load(&model, model_path) != 0) {
        fprintf(stderr, "Fatal: failed to load model\n");
        return 1;
    }

    /* ── Load Tokenizer ── */
    fprintf(stderr, "Loading tokenizer...\n");
    Tokenizer tok = {0};
    if (load_tokenizer_from_model(&tok, model_path) != 0) {
        fprintf(stderr, "Fatal: failed to load tokenizer\n");
        model_free(&model);
        return 1;
    }
    fprintf(stderr, "Tokenizer: %d vocab, %d merges\n\n",
            tok.vocab_size, tok.n_merges);

    /* ── Allocate RunState ── */
    RunState state = {0};
    if (runstate_alloc(&state, &model.config) != 0) {
        fprintf(stderr, "Fatal: failed to allocate run state\n");
        tokenizer_free(&tok);
        model_free(&model);
        return 1;
    }

    /* Memory usage estimate */
    {
        size_t kv_mem = (size_t)model.config.n_layers * 2 *
                        model.config.max_seq_len * model.config.n_heads *
                        model.config.head_dim * sizeof(float);
        fprintf(stderr, "KV Cache memory: %.1f MB\n", (double)kv_mem / 1e6);
    }

    /* ── Run ── */
    if (interactive) {
        interactive_mode(&model, &state, &tok, temperature, top_k, top_p);
    } else if (prompt) {
        generate(&model, &state, &tok, prompt, max_tokens,
                 temperature, top_k, top_p, use_reasoning);
    } else {
        fprintf(stderr, "Error: provide a prompt or use --interactive\n");
        print_usage(argv[0]);
    }

    /* ── Cleanup ── */
    runstate_free(&state);
    tokenizer_free(&tok);
    model_free(&model);

    return 0;
}
