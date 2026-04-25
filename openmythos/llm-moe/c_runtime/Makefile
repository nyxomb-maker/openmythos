# ═══════════════════════════════════════════════════════════════════════
# LLM-MoE C Runtime — Makefile
# ═══════════════════════════════════════════════════════════════════════
#
# Targets:
#   make           — build optimized release binary
#   make debug     — build with debug symbols and sanitizers
#   make clean     — remove build artifacts
#   make openmp    — build with OpenMP parallelization
#
# No external dependencies. Pure C99.
# ═══════════════════════════════════════════════════════════════════════

CC       = gcc
TARGET   = llm-moe

# Source files
SRCS     = main.c model.c attention.c moe.c math_ops.c quantize.c \
           tokenizer.c reasoning.c

OBJS     = $(SRCS:.c=.o)

# ── Flags ────────────────────────────────────────────────────────────

CFLAGS_COMMON  = -std=c99 -Wall -Wextra -Wpedantic \
                 -Wno-unused-parameter -Wno-sign-compare \
                 -D_POSIX_C_SOURCE=199309L

CFLAGS_RELEASE = $(CFLAGS_COMMON) -O3 -march=native -ffast-math \
                 -funroll-loops -DNDEBUG

CFLAGS_DEBUG   = $(CFLAGS_COMMON) -O0 -g3 -fsanitize=address,undefined \
                 -fno-omit-frame-pointer

CFLAGS_OMP     = $(CFLAGS_RELEASE) -fopenmp

LDFLAGS        = -lm
LDFLAGS_DEBUG  = -lm -fsanitize=address,undefined

# ── Default: Release build ──────────────────────────────────────────

all: CFLAGS = $(CFLAGS_RELEASE)
all: LDLIBS = $(LDFLAGS)
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	@echo ""
	@echo "  ✓ Built $(TARGET) (release, optimized)"
	@echo ""

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Debug build ─────────────────────────────────────────────────────

debug: CFLAGS = $(CFLAGS_DEBUG)
debug: LDLIBS = $(LDFLAGS_DEBUG)
debug: clean $(TARGET)
	@echo ""
	@echo "  ✓ Built $(TARGET) (debug, sanitizers enabled)"
	@echo ""

# ── OpenMP build ────────────────────────────────────────────────────

openmp: CFLAGS = $(CFLAGS_OMP)
openmp: LDLIBS = $(LDFLAGS) -fopenmp
openmp: clean $(TARGET)
	@echo ""
	@echo "  ✓ Built $(TARGET) (release + OpenMP)"
	@echo ""

# ── Clean ───────────────────────────────────────────────────────────

clean:
	rm -f $(OBJS) $(TARGET)
	@echo "  ✓ Cleaned"

# ── Phony targets ──────────────────────────────────────────────────

.PHONY: all debug openmp clean
