/*
 * Training Data — C Language Examples
 * Systems programming patterns, memory management, data structures.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Dynamic Array (Vector)
 * ═══════════════════════════════════════════════════════════════════════*/

typedef struct {
    int *data;
    size_t size;
    size_t capacity;
} Vector;

Vector *vector_create(size_t initial_capacity) {
    Vector *v = (Vector *)malloc(sizeof(Vector));
    if (!v) return NULL;
    v->data = (int *)malloc(initial_capacity * sizeof(int));
    if (!v->data) { free(v); return NULL; }
    v->size = 0;
    v->capacity = initial_capacity;
    return v;
}

void vector_push(Vector *v, int value) {
    if (v->size >= v->capacity) {
        v->capacity *= 2;
        int *new_data = (int *)realloc(v->data, v->capacity * sizeof(int));
        if (!new_data) {
            fprintf(stderr, "Error: realloc failed\n");
            return;
        }
        v->data = new_data;
    }
    v->data[v->size++] = value;
}

int vector_pop(Vector *v) {
    if (v->size == 0) {
        fprintf(stderr, "Error: pop from empty vector\n");
        return -1;
    }
    return v->data[--v->size];
}

int vector_get(const Vector *v, size_t index) {
    if (index >= v->size) {
        fprintf(stderr, "Error: index out of bounds\n");
        return -1;
    }
    return v->data[index];
}

void vector_free(Vector *v) {
    if (v) {
        free(v->data);
        free(v);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Hash Map (Open Addressing)
 * ═══════════════════════════════════════════════════════════════════════*/

#define HASHMAP_INITIAL_SIZE 64
#define HASHMAP_LOAD_FACTOR 0.75

typedef struct {
    char *key;
    int value;
    int occupied;
} HashEntry;

typedef struct {
    HashEntry *entries;
    size_t capacity;
    size_t size;
} HashMap;

static uint32_t fnv1a_hash(const char *key) {
    uint32_t hash = 2166136261u;
    while (*key) {
        hash ^= (uint8_t)*key++;
        hash *= 16777619u;
    }
    return hash;
}

HashMap *hashmap_create(void) {
    HashMap *map = (HashMap *)malloc(sizeof(HashMap));
    if (!map) return NULL;
    map->capacity = HASHMAP_INITIAL_SIZE;
    map->size = 0;
    map->entries = (HashEntry *)calloc(map->capacity, sizeof(HashEntry));
    if (!map->entries) { free(map); return NULL; }
    return map;
}

void hashmap_put(HashMap *map, const char *key, int value) {
    /* Check load factor */
    if ((double)map->size / map->capacity > HASHMAP_LOAD_FACTOR) {
        /* Resize: double capacity and rehash */
        size_t old_cap = map->capacity;
        HashEntry *old_entries = map->entries;

        map->capacity *= 2;
        map->entries = (HashEntry *)calloc(map->capacity, sizeof(HashEntry));
        map->size = 0;

        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i].occupied) {
                hashmap_put(map, old_entries[i].key, old_entries[i].value);
                free(old_entries[i].key);
            }
        }
        free(old_entries);
    }

    uint32_t idx = fnv1a_hash(key) % map->capacity;

    /* Linear probing */
    while (map->entries[idx].occupied) {
        if (strcmp(map->entries[idx].key, key) == 0) {
            map->entries[idx].value = value;
            return;
        }
        idx = (idx + 1) % map->capacity;
    }

    map->entries[idx].key = strdup(key);
    map->entries[idx].value = value;
    map->entries[idx].occupied = 1;
    map->size++;
}

int hashmap_get(const HashMap *map, const char *key, int *out_value) {
    uint32_t idx = fnv1a_hash(key) % map->capacity;
    size_t start = idx;

    do {
        if (!map->entries[idx].occupied) return 0;
        if (strcmp(map->entries[idx].key, key) == 0) {
            *out_value = map->entries[idx].value;
            return 1;
        }
        idx = (idx + 1) % map->capacity;
    } while (idx != start);

    return 0;
}

void hashmap_free(HashMap *map) {
    if (map) {
        for (size_t i = 0; i < map->capacity; i++) {
            if (map->entries[i].occupied) {
                free(map->entries[i].key);
            }
        }
        free(map->entries);
        free(map);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Memory Pool Allocator
 * ═══════════════════════════════════════════════════════════════════════*/

typedef struct PoolBlock {
    struct PoolBlock *next;
} PoolBlock;

typedef struct {
    void *memory;
    PoolBlock *free_list;
    size_t block_size;
    size_t total_blocks;
} MemoryPool;

MemoryPool *pool_create(size_t block_size, size_t num_blocks) {
    if (block_size < sizeof(PoolBlock))
        block_size = sizeof(PoolBlock);

    MemoryPool *pool = (MemoryPool *)malloc(sizeof(MemoryPool));
    if (!pool) return NULL;

    pool->memory = malloc(block_size * num_blocks);
    if (!pool->memory) { free(pool); return NULL; }

    pool->block_size = block_size;
    pool->total_blocks = num_blocks;

    /* Build free list */
    pool->free_list = NULL;
    uint8_t *ptr = (uint8_t *)pool->memory;
    for (size_t i = 0; i < num_blocks; i++) {
        PoolBlock *block = (PoolBlock *)(ptr + i * block_size);
        block->next = pool->free_list;
        pool->free_list = block;
    }

    return pool;
}

void *pool_alloc(MemoryPool *pool) {
    if (!pool->free_list) return NULL;
    PoolBlock *block = pool->free_list;
    pool->free_list = block->next;
    return block;
}

void pool_free_block(MemoryPool *pool, void *ptr) {
    PoolBlock *block = (PoolBlock *)ptr;
    block->next = pool->free_list;
    pool->free_list = block;
}

void pool_destroy(MemoryPool *pool) {
    if (pool) {
        free(pool->memory);
        free(pool);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Ring Buffer (Lock-Free Style)
 * ═══════════════════════════════════════════════════════════════════════*/

typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t head;  /* write position */
    size_t tail;  /* read position */
} RingBuffer;

RingBuffer *ring_create(size_t capacity) {
    RingBuffer *rb = (RingBuffer *)malloc(sizeof(RingBuffer));
    if (!rb) return NULL;
    rb->buffer = (uint8_t *)malloc(capacity);
    if (!rb->buffer) { free(rb); return NULL; }
    rb->capacity = capacity;
    rb->head = rb->tail = 0;
    return rb;
}

int ring_write(RingBuffer *rb, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        size_t next_head = (rb->head + 1) % rb->capacity;
        if (next_head == rb->tail) return -1;  /* full */
        rb->buffer[rb->head] = data[i];
        rb->head = next_head;
    }
    return 0;
}

int ring_read(RingBuffer *rb, uint8_t *out, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (rb->tail == rb->head) return -1;  /* empty */
        out[i] = rb->buffer[rb->tail];
        rb->tail = (rb->tail + 1) % rb->capacity;
    }
    return 0;
}

void ring_free(RingBuffer *rb) {
    if (rb) {
        free(rb->buffer);
        free(rb);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Binary Search Variants
 * ═══════════════════════════════════════════════════════════════════════*/

int binary_search(const int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int lower_bound(const int *arr, int n, int target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int upper_bound(const int *arr, int n, int target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
