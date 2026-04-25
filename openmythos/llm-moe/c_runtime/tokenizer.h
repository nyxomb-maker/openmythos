/*
 * tokenizer.h — BPE tokenizer for C runtime
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum token string length */
#define MAX_TOKEN_LEN 256

/* ── Merge rule ─────────────────────────────────────────────────────── */
typedef struct {
    char  a[MAX_TOKEN_LEN];
    char  b[MAX_TOKEN_LEN];
    int   a_id;    /* cached vocab ID of a, or -1 */
    int   b_id;    /* cached vocab ID of b, or -1 */
} MergeRule;

/* ── Tokenizer ──────────────────────────────────────────────────────── */
typedef struct {
    /* Vocabulary: token_id → string */
    char   **vocab;        /* array of token strings */
    int      vocab_size;

    /* Merge rules (ordered by priority) */
    MergeRule *merges;
    int        n_merges;
} Tokenizer;

/* Load tokenizer from model binary file.
 * Reads from file pointer at current position.
 * Returns 0 on success.
 */
int tokenizer_load(Tokenizer *tok, const char *filepath);

/* Load tokenizer from already-open FILE at current position.
 * tok_vocab_size and tok_n_merges should already be read from header.
 */
int tokenizer_load_from_fp(Tokenizer *tok, void *fp,
                           int vocab_size, int n_merges);

/* Free tokenizer memory. */
void tokenizer_free(Tokenizer *tok);

/* Encode text to token IDs.
 * Returns number of tokens written to `out`.
 * `out` must have space for at least `max_tokens` IDs.
 * If add_special is non-zero, prepends BOS and appends EOS.
 */
int tokenizer_encode(const Tokenizer *tok, const char *text,
                     int *out, int max_tokens, int add_special,
                     int bos_id, int eos_id);

/* Decode a single token ID to its string representation.
 * Returns pointer to the token string (owned by tokenizer).
 * Returns "<unk>" for invalid IDs.
 */
const char *tokenizer_decode(const Tokenizer *tok, int token_id);

/* Look up a string in the vocabulary.
 * Returns token ID or -1 if not found.
 */
int tokenizer_lookup(const Tokenizer *tok, const char *str);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */
