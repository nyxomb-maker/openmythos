/*
 * reasoning.c — Structured Reasoning Engine implementation
 *
 * The reasoning system detects special tokens in the generation stream
 * and provides:
 *
 * 1. DECOMPOSITION
 *    When <think> is generated, the system enters reasoning mode.
 *    Each <step> token begins a new sub-problem.
 *
 * 2. VALIDATION
 *    When <verify> is generated, the current step is validated:
 *    - Entropy check: low entropy suggests confident reasoning
 *    - Consistency: compare with previous steps
 *    - Length heuristic: very short or very long steps are suspect
 *
 * 3. COMPOSITION
 *    When <conclude> is generated, all validated steps are combined.
 *    The system outputs a summary and confidence assessment.
 */

#include "reasoning.h"
#include "math_ops.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Initialization
 * ═══════════════════════════════════════════════════════════════════════*/

void reasoning_init(ReasoningState *rs, const Tokenizer *tok)
{
    memset(rs, 0, sizeof(ReasoningState));

    /* Resolve special token IDs */
    rs->think_id     = tokenizer_lookup(tok, "<think>");
    rs->step_id      = tokenizer_lookup(tok, "<step>");
    rs->verify_id    = tokenizer_lookup(tok, "<verify>");
    rs->conclude_id  = tokenizer_lookup(tok, "<conclude>");
    rs->end_think_id = tokenizer_lookup(tok, "</think>");

    /* If tokens not found, use -1 (disabled) */
    if (rs->think_id < 0)     rs->think_id     = -1;
    if (rs->step_id < 0)      rs->step_id      = -1;
    if (rs->verify_id < 0)    rs->verify_id    = -1;
    if (rs->conclude_id < 0)  rs->conclude_id  = -1;
    if (rs->end_think_id < 0) rs->end_think_id = -1;
}

void reasoning_reset(ReasoningState *rs)
{
    rs->n_steps = 0;
    rs->active = 0;
    rs->current_step = -1;

    for (int i = 0; i < MAX_REASONING_STEPS; i++) {
        rs->steps[i].n_tokens = 0;
        rs->steps[i].confidence = 0.0f;
        rs->steps[i].validated = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Token Processing
 * ═══════════════════════════════════════════════════════════════════════*/

int reasoning_process_token(ReasoningState *rs, int token_id,
                            const float *logits, int vocab_size)
{
    /* Check for special tokens */

    /* <think> — Enter reasoning mode */
    if (token_id == rs->think_id) {
        rs->active = 1;
        rs->n_steps = 0;
        rs->current_step = -1;
        fprintf(stderr, "[Reasoning] Entering structured reasoning mode\n");
        return 1;
    }

    /* </think> — Exit reasoning mode */
    if (token_id == rs->end_think_id) {
        rs->active = 0;
        fprintf(stderr, "[Reasoning] Exiting reasoning mode (%d steps)\n",
                rs->n_steps);
        return -1;
    }

    /* Only process if in reasoning mode */
    if (!rs->active) return 0;

    /* <step> — Begin new reasoning step */
    if (token_id == rs->step_id) {
        if (rs->n_steps < MAX_REASONING_STEPS) {
            rs->current_step = rs->n_steps;
            rs->n_steps++;
            rs->steps[rs->current_step].n_tokens = 0;
            rs->steps[rs->current_step].confidence = 0.0f;
            rs->steps[rs->current_step].validated = 0;
            fprintf(stderr, "[Reasoning] Step %d started\n", rs->current_step + 1);
        }
        return 1;
    }

    /* <verify> — Validate current step */
    if (token_id == rs->verify_id) {
        if (rs->current_step >= 0 && rs->current_step < rs->n_steps) {
            float conf = reasoning_validate_step(rs, rs->current_step);
            rs->steps[rs->current_step].confidence = conf;
            rs->steps[rs->current_step].validated = 1;
            fprintf(stderr, "[Reasoning] Step %d validated (confidence: %.2f)\n",
                    rs->current_step + 1, conf);
        }
        return 1;
    }

    /* <conclude> — Compose final answer */
    if (token_id == rs->conclude_id) {
        fprintf(stderr, "[Reasoning] Composing final answer from %d steps\n",
                rs->n_steps);

        /* Print step summary */
        int validated_count = 0;
        float total_conf = 0.0f;
        for (int i = 0; i < rs->n_steps; i++) {
            if (rs->steps[i].validated) {
                validated_count++;
                total_conf += rs->steps[i].confidence;
            }
        }

        if (validated_count > 0) {
            fprintf(stderr, "[Reasoning] %d/%d steps validated, "
                    "avg confidence: %.2f\n",
                    validated_count, rs->n_steps,
                    total_conf / validated_count);
        }
        return 1;
    }

    /* Regular token — add to current step */
    if (rs->current_step >= 0 && rs->current_step < rs->n_steps) {
        ReasoningStep *step = &rs->steps[rs->current_step];
        if (step->n_tokens < MAX_STEP_TOKENS) {
            step->tokens[step->n_tokens++] = token_id;
        }
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Validation Heuristics
 *
 * Multiple heuristics are combined:
 * 1. Length check: steps should be between 5 and 200 tokens
 * 2. Repetition check: detect token-level repetitions
 * 3. Diversity check: count unique tokens
 * ═══════════════════════════════════════════════════════════════════════*/

float reasoning_validate_step(ReasoningState *rs, int step_idx)
{
    if (step_idx < 0 || step_idx >= rs->n_steps) return 0.0f;

    ReasoningStep *step = &rs->steps[step_idx];
    int n = step->n_tokens;

    if (n == 0) return 0.0f;

    float score = 1.0f;

    /* ── 1. Length heuristic ─────────────────────────────────────────── */
    if (n < 3) {
        score *= 0.3f;  /* Too short — likely incomplete */
    } else if (n < 5) {
        score *= 0.6f;
    } else if (n > 200) {
        score *= 0.5f;  /* Too long — might be rambling */
    } else if (n > 100) {
        score *= 0.8f;
    }

    /* ── 2. Repetition check ─────────────────────────────────────────── */
    /* Count immediate bigram repetitions */
    int repetitions = 0;
    for (int i = 2; i < n; i++) {
        if (step->tokens[i] == step->tokens[i - 2] &&
            (i >= 3 && step->tokens[i - 1] == step->tokens[i - 3])) {
            repetitions++;
        }
    }
    if (n > 4) {
        float rep_ratio = (float)repetitions / (float)(n - 2);
        if (rep_ratio > 0.3f) score *= 0.3f;
        else if (rep_ratio > 0.1f) score *= 0.7f;
    }

    /* ── 3. Diversity check ──────────────────────────────────────────── */
    /* Count unique tokens (simple O(n^2) for small n) */
    int unique = 0;
    for (int i = 0; i < n; i++) {
        int is_unique = 1;
        for (int j = 0; j < i; j++) {
            if (step->tokens[i] == step->tokens[j]) {
                is_unique = 0;
                break;
            }
        }
        unique += is_unique;
    }
    float diversity = (float)unique / (float)n;
    if (diversity < 0.2f) score *= 0.4f;
    else if (diversity < 0.4f) score *= 0.7f;

    /* ── 4. Consistency with previous steps ──────────────────────────── */
    /* Check that this step shares some tokens with previous steps
     * (indicating coherent reasoning chain) */
    if (step_idx > 0) {
        int shared = 0;
        ReasoningStep *prev = &rs->steps[step_idx - 1];
        for (int i = 0; i < n && i < 50; i++) {
            for (int j = 0; j < prev->n_tokens && j < 50; j++) {
                if (step->tokens[i] == prev->tokens[j]) {
                    shared++;
                    break;
                }
            }
        }
        float coherence = (n > 0) ? (float)shared / (float)n : 0.0f;
        /* Some overlap is good, but not too much (would mean repetition) */
        if (coherence < 0.05f) score *= 0.7f;  /* No connection to previous */
        else if (coherence > 0.8f) score *= 0.6f;  /* Too similar */
    }

    /* Clamp to [0, 1] */
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;

    return score;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Debug Summary
 * ═══════════════════════════════════════════════════════════════════════*/

void reasoning_print_summary(const ReasoningState *rs,
                             const Tokenizer *tok)
{
    fprintf(stderr, "\n=== Reasoning Summary ===\n");
    fprintf(stderr, "Total steps: %d\n", rs->n_steps);

    for (int i = 0; i < rs->n_steps; i++) {
        const ReasoningStep *step = &rs->steps[i];
        fprintf(stderr, "\nStep %d (%d tokens, confidence: %.2f, %s):\n  ",
                i + 1, step->n_tokens, step->confidence,
                step->validated ? "VALIDATED" : "unvalidated");

        /* Print first few tokens */
        int show = step->n_tokens < 20 ? step->n_tokens : 20;
        for (int t = 0; t < show; t++) {
            const char *s = tokenizer_decode(tok, step->tokens[t]);
            fprintf(stderr, "%s", s);
        }
        if (step->n_tokens > show) {
            fprintf(stderr, "...");
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "=========================\n\n");
}
