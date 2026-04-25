/*
 * reasoning.h — Structured Reasoning Engine
 *
 * Provides multi-step reasoning with:
 * - Decomposition: break complex problems into steps
 * - Validation: verify each step before proceeding
 * - Composition: combine partial results into final answer
 *
 * Uses special tokens:
 *   <think>   — begin reasoning mode
 *   <step>    — start a sub-step
 *   <verify>  — trigger verification
 *   <conclude>— combine results
 *   </think>  — end reasoning mode
 */

#ifndef REASONING_H
#define REASONING_H

#include "model.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum reasoning steps */
#define MAX_REASONING_STEPS 32

/* Maximum tokens per step */
#define MAX_STEP_TOKENS 256

/* ── Reasoning Step ─────────────────────────────────────────────────── */
typedef struct {
    int   tokens[MAX_STEP_TOKENS];
    int   n_tokens;
    float confidence;       /* heuristic confidence score [0, 1] */
    int   validated;        /* 1 if step passed validation */
} ReasoningStep;

/* ── Reasoning State ────────────────────────────────────────────────── */
typedef struct {
    ReasoningStep steps[MAX_REASONING_STEPS];
    int           n_steps;
    int           active;           /* 1 if currently in reasoning mode */
    int           current_step;     /* index of current step being generated */

    /* Special token IDs (resolved at init) */
    int think_id;
    int step_id;
    int verify_id;
    int conclude_id;
    int end_think_id;
} ReasoningState;

/* Initialize reasoning state and resolve special token IDs. */
void reasoning_init(ReasoningState *rs, const Tokenizer *tok);

/* Reset reasoning state for a new query. */
void reasoning_reset(ReasoningState *rs);

/* Process a newly generated token through the reasoning engine.
 * Returns:
 *   0: normal token, continue generation
 *   1: reasoning state changed (new step, verification, etc.)
 *  -1: reasoning complete (</think> encountered)
 */
int reasoning_process_token(ReasoningState *rs, int token_id,
                            const float *logits, int vocab_size);

/* Validate the current step using heuristic checks.
 * Returns confidence score [0, 1].
 */
float reasoning_validate_step(ReasoningState *rs, int step_idx);

/* Get a summary of the reasoning process for debugging. */
void reasoning_print_summary(const ReasoningState *rs,
                             const Tokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif /* REASONING_H */
