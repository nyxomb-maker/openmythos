/*
 * moe.h — Mixture of Experts layer
 */

#ifndef MOE_H
#define MOE_H

#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Forward pass through MoE layer for a single token.
 *
 * 1. Compute gating logits: gate(x) → [n_experts]
 * 2. Softmax → probabilities
 * 3. Select top-K experts
 * 4. Compute each selected expert's output (SwiGLU MLP)
 * 5. Weighted combination → output
 *
 * Args:
 *   out:    output buffer [dim]
 *   input:  input hidden state [dim]
 *   moe:    MoE layer weights
 *   state:  RunState with scratch buffers
 *   config: model config
 */
void moe_forward(float *out, const float *input,
                 const MoELayer *moe, RunState *state,
                 const Config *config);

#ifdef __cplusplus
}
#endif

#endif /* MOE_H */
