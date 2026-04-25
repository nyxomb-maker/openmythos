/*
 * tokenizer.c — BPE tokenizer in C
 *
 * Implements:
 *   - Loading vocab + merges from model binary
 *   - Encoding text to token IDs (BPE merge algorithm)
 *   - Decoding token IDs to text
 *   - No external dependencies
 */

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Loading
 * ═══════════════════════════════════════════════════════════════════════*/

int tokenizer_load_from_fp(Tokenizer *tok, void *fp_void,
                           int vocab_size, int n_merges)
{
    FILE *fp = (FILE *)fp_void;

    tok->vocab_size = vocab_size;
    tok->n_merges = n_merges;

    /* Allocate vocab array */
    tok->vocab = (char **)calloc((size_t)vocab_size, sizeof(char *));
    if (!tok->vocab) return -1;

    /* Read vocab entries */
    for (int i = 0; i < vocab_size; i++) {
        uint16_t len;
        if (fread(&len, sizeof(uint16_t), 1, fp) != 1) return -1;

        char *str = (char *)malloc((size_t)len + 1);
        if (!str) return -1;

        if (len > 0) {
            if (fread(str, 1, (size_t)len, fp) != (size_t)len) {
                free(str);
                return -1;
            }
        }
        str[len] = '\0';

        uint32_t token_id;
        if (fread(&token_id, sizeof(uint32_t), 1, fp) != 1) {
            free(str);
            return -1;
        }

        if ((int)token_id < vocab_size) {
            tok->vocab[token_id] = str;
        } else {
            free(str);
        }
    }

    /* Read merge rules */
    tok->merges = (MergeRule *)calloc((size_t)n_merges, sizeof(MergeRule));
    if (!tok->merges) return -1;

    for (int i = 0; i < n_merges; i++) {
        uint16_t a_len, b_len;

        if (fread(&a_len, sizeof(uint16_t), 1, fp) != 1) return -1;
        if (a_len >= MAX_TOKEN_LEN) return -1;
        if (fread(tok->merges[i].a, 1, (size_t)a_len, fp) != (size_t)a_len) return -1;
        tok->merges[i].a[a_len] = '\0';

        if (fread(&b_len, sizeof(uint16_t), 1, fp) != 1) return -1;
        if (b_len >= MAX_TOKEN_LEN) return -1;
        if (fread(tok->merges[i].b, 1, (size_t)b_len, fp) != (size_t)b_len) return -1;
        tok->merges[i].b[b_len] = '\0';

        /* Cache vocab lookups */
        tok->merges[i].a_id = tokenizer_lookup(tok, tok->merges[i].a);
        tok->merges[i].b_id = tokenizer_lookup(tok, tok->merges[i].b);
    }

    return 0;
}

int tokenizer_load(Tokenizer *tok, const char *filepath)
{
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;

    /* Skip magic + version + config */
    /* This function expects the file to be positioned at the tokenizer section */
    /* For standalone loading, we'd need to skip the header first */

    uint32_t vocab_size, n_merges;
    if (fread(&vocab_size, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&n_merges, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return -1; }

    int ret = tokenizer_load_from_fp(tok, fp, (int)vocab_size, (int)n_merges);
    fclose(fp);
    return ret;
}

void tokenizer_free(Tokenizer *tok)
{
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
        tok->vocab = NULL;
    }
    free(tok->merges);
    tok->merges = NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Lookup
 * ═══════════════════════════════════════════════════════════════════════*/

int tokenizer_lookup(const Tokenizer *tok, const char *str)
{
    /* Linear scan — adequate for small vocabs.
     * For large vocabs, a hash table would be faster. */
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i] && strcmp(tok->vocab[i], str) == 0) {
            return i;
        }
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Encoding
 *
 * Algorithm:
 *   1. Convert input text to byte-level tokens
 *   2. Iteratively apply learned merges in priority order
 *   3. Map resulting tokens to IDs
 * ═══════════════════════════════════════════════════════════════════════*/

/* Convert a single byte to its token string representation */
static void byte_to_token_str(unsigned char b, char *out)
{
    if (b >= 33 && b <= 126 && b != '<' && b != '>') {
        out[0] = (char)b;
        out[1] = '\0';
    } else {
        snprintf(out, 16, "<0x%02X>", b);
    }
}

int tokenizer_encode(const Tokenizer *tok, const char *text,
                     int *out, int max_tokens, int add_special,
                     int bos_id, int eos_id)
{
    if (!text || !out || max_tokens <= 0) return 0;

    /* Working buffer: array of token strings */
    int text_len = (int)strlen(text);
    if (text_len == 0) {
        int n = 0;
        if (add_special && n < max_tokens) out[n++] = bos_id;
        if (add_special && n < max_tokens) out[n++] = eos_id;
        return n;
    }

    /* Allocate working arrays */
    int capacity = text_len + 2;
    char **tokens = (char **)malloc((size_t)capacity * sizeof(char *));
    int n_tokens = 0;

    if (!tokens) return 0;

    /* Convert each byte to a token string */
    const unsigned char *bytes = (const unsigned char *)text;
    for (int i = 0; i < text_len; i++) {
        char *t = (char *)malloc(16);
        if (!t) break;
        byte_to_token_str(bytes[i], t);
        tokens[n_tokens++] = t;
    }

    /* Apply merges iteratively */
    for (int m = 0; m < tok->n_merges && n_tokens > 1; m++) {
        const char *a = tok->merges[m].a;
        const char *b = tok->merges[m].b;

        /* Scan for this merge pair */
        int found = 0;
        for (int i = 0; i < n_tokens - 1; i++) {
            if (strcmp(tokens[i], a) == 0 && strcmp(tokens[i + 1], b) == 0) {
                /* Merge: concatenate a+b */
                int a_len = (int)strlen(tokens[i]);
                int b_len = (int)strlen(tokens[i + 1]);
                char *merged = (char *)malloc((size_t)(a_len + b_len + 1));
                if (!merged) continue;

                memcpy(merged, tokens[i], (size_t)a_len);
                memcpy(merged + a_len, tokens[i + 1], (size_t)b_len);
                merged[a_len + b_len] = '\0';

                /* Replace tokens[i] with merged, remove tokens[i+1] */
                free(tokens[i]);
                free(tokens[i + 1]);
                tokens[i] = merged;

                /* Shift remaining tokens left */
                for (int j = i + 1; j < n_tokens - 1; j++) {
                    tokens[j] = tokens[j + 1];
                }
                n_tokens--;

                found = 1;
                i--;  /* Check same position again (might merge again) */
            }
        }
        (void)found;
    }

    /* Convert token strings to IDs */
    int n_out = 0;

    if (add_special && n_out < max_tokens) {
        out[n_out++] = bos_id;
    }

    for (int i = 0; i < n_tokens && n_out < max_tokens; i++) {
        int id = tokenizer_lookup(tok, tokens[i]);
        out[n_out++] = (id >= 0) ? id : 1;  /* 1 = <unk> */
    }

    if (add_special && n_out < max_tokens) {
        out[n_out++] = eos_id;
    }

    /* Cleanup */
    for (int i = 0; i < n_tokens; i++) {
        free(tokens[i]);
    }
    free(tokens);

    return n_out;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Decoding
 * ═══════════════════════════════════════════════════════════════════════*/

const char *tokenizer_decode(const Tokenizer *tok, int token_id)
{
    if (token_id < 0 || token_id >= tok->vocab_size || !tok->vocab[token_id]) {
        return "<unk>";
    }
    return tok->vocab[token_id];
}
