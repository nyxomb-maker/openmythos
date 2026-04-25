/*
 * attention.h — Causal self-attention with KV cache and Flash Attention
 */

#ifndef ATTENTION_H
#define ATTENTION_H

#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Perform causal self-attention for a single token at position `pos`.
 *
 * Implements Flash Attention in streaming mode:
 *   - Processes keys/values in blocks
 *   - Uses online softmax for numerical stability
 *   - Never materializes full N×N attention matrix
 *   - Memory: O(block_size) instead of O(N)
 *
 * Args:
 *   state:  RunState with q/k/v/att buffers
 *   cache:  KVCache for this layer (updated in-place)
 *   config: model config
 *   pos:    current token position
 */
void attention_forward(RunState *state, KVCache *cache,
                       const Config *config, int pos);

/*
 * Standard (non-flash) attention for reference/validation.
 * Materializes full attention scores — O(N) memory per head.
 */
void attention_forward_standard(RunState *state, KVCache *cache,
                                const Config *config, int pos);

#ifdef __cplusplus
}
#endif

#endif /* ATTENTION_H */
