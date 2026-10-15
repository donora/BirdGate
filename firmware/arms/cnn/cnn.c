/* cnn.c — the CNN arm's scorer: three 3x3 convolutions, streamed one log-mel column
 * at a time, global average pool, one linear unit.
 *
 * Mirrors two Python references at once, and must match both:
 *
 *   cnn.stream_forward  — the streaming SCHEDULE. Each stage owns its own time
 *       padding, which is the subtlety: `padding=1` is a boundary condition on
 *       EVERY layer's input, not just the network's. docs/lab-notebook.md §9c records what that
 *       costs to get wrong — every interior column bit-exact and only the two edge
 *       columns adrift, which no eyeball catches.
 *   cnn.QuantGate      — the ARITHMETIC. A quantiser sits after every module, so the
 *       order is conv, quantise, ReLU, quantise, pool, quantise, and the scales
 *       differ at each point.
 *
 * `cnn.stream_forward_quant` is the reference that combines them, and
 * `tests/test_c_parity.py` pins this file against it.
 *
 * **Why int8 storage.** Held as float, the stage rings are 9,600 B and the arm does
 * not fit beside stock firmware — the 2.4 KB figure in docs/lab-notebook.md §12 was always an
 * int8 number. Held as one byte per activation they are 2,400 B and it fits.
 * Arithmetic stays in float on the FPU: a code is read back as
 * `(code - offset) * scale`, which is exactly the value `ActQuant` produces, so
 * this is the int8 grid without CMSIS-NN's integer requantisation. That is why the
 * existing `cnn-int8` numbers describe this implementation rather than an
 * approximation of it.
 */
#include "arm.h"
#include "logmel.h"
#include <math.h>
#include <string.h>

/* Activation quantisers, in the order cnn.quantise builds them: the input, then one
 * after every module. Blocks are (conv, ReLU, pool) x BG_N_CONV, then the average
 * pool, the flatten and the linear unit. */
#define Q_IN            0
#define Q_CONV(i)       (1 + 3 * (i))
#define Q_RELU(i)       (2 + 3 * (i))
#define Q_POOL(i)       (3 + 3 * (i))
#define Q_AVG           (1 + 3 * BG_N_CONV)
#define Q_FLAT          (Q_AVG + 1)
#define Q_LIN           (Q_AVG + 2)

/* The last two are pass-through, and the exporter marks them with a zero scale.
 * `Flatten` changes no values, and the single logit goes straight to a comparison
 * rather than into a byte — on a device it is read out of the accumulator. Both are
 * artefacts of quantising after every module, and the linear one was coarse enough
 * (a step of ~0.07 against a threshold near 0.54) to move gate decisions. */
typedef char bg_act_table_matches_network[(BG_N_ACT == Q_LIN + 1) ? 1 : -1];  /* C99 */
typedef char bg_kernel_is_three[(BG_CONV0_K == 3 && BG_CONV1_K == 3 && BG_CONV2_K == 3) ? 1 : -1];

/* One code back to the value ActQuant would have produced. */
static inline float deq(uint8_t code, int q) {
    return ((float)code - BG_ACT_OFFSET[q]) * BG_ACT_SCALE[q];
}

/* round-half-away-from-zero without the library call. `roundf` is an out-of-line
 * function in newlib and this runs ~49,000 times per window; a cast truncates toward
 * zero, so adding a half before it gives the same answer in two instructions. Exact
 * for every value that reaches here — the result is clamped into a byte immediately
 * after — and the parity numbers are unchanged to the digit, which is the check that
 * matters rather than the argument. */
static inline float bg_roundf(float v) {
    return v >= 0.0f ? (float)(int32_t)(v + 0.5f) : -(float)(int32_t)(0.5f - v);
}

/* Quantise onto an activation grid, then clamp into the byte. */
static inline uint8_t quant(float v, int q) {
    float r = bg_roundf(v / BG_ACT_SCALE[q]) + BG_ACT_OFFSET[q];
    if (r < 0.0f) r = 0.0f;
    if (r > 255.0f) r = 255.0f;
    return (uint8_t)r;
}

/* --- the cascade ----------------------------------------------------------------
 * Stage 2i is conv i, stage 2i+1 is the pool after it; `layer_spec` in Python emits
 * them in exactly this order. A stage holds the last BG_STAGE_K columns of its own
 * INPUT, so the rings are sized by the input side.
 */
#define N_STAGE        (2 * BG_N_CONV)
#define MAX_COL        (BG_CONV0_OUT * BG_N_MELS)   /* widest column: after conv 0 */

typedef struct {
    uint8_t  kind;          /* 0 = conv, 1 = pool */
    uint8_t  idx;           /* which conv, or which pool */
    uint8_t  k, s, p;
    uint16_t ch, fq;        /* channels and frequency bins of its INPUT column */
    uint16_t n_in, n_out;
    uint16_t seen, next_out, filled;
    uint8_t *ring;          /* k columns of ch*fq codes */
    int      q_in;          /* quantiser whose grid the ring holds */
} stage_t;

static uint8_t ring0[3 * BG_CONV0_IN  * BG_N_MELS];
static uint8_t ring1[2 * BG_CONV0_OUT * BG_N_MELS];
static uint8_t ring2[3 * BG_CONV1_IN  * (BG_N_MELS / 2)];
static uint8_t ring3[2 * BG_CONV1_OUT * (BG_N_MELS / 2)];
static uint8_t ring4[3 * BG_CONV2_IN  * (BG_N_MELS / 4)];
static uint8_t ring5[2 * BG_CONV2_OUT * (BG_N_MELS / 4)];

static stage_t stages[N_STAGE];

/* Causal estimate of the acquisition level, mirroring cnn.level_centres.
 *
 * The arm was level-dependent without it: the input quantiser spans about 46 dB, so a
 * device quieter than the training corpus clamps its input and the network goes blind.
 * Measured in the simulator, detection falls from 91.5% to 19.0% at -20 dB and to
 * nothing at -30 dB, while the linear arm is unaffected.
 *
 * One update per window, from that window's own mean, applied to the NEXT window, so
 * nothing depends on audio that has not been processed. Per frame would be the obvious
 * choice and is not reproducible here: only 97 frames of every 100 are ever computed. */
static float    level;            /* the tracked level, carried across windows */
static float    window_centre;    /* what THIS window is normalised by */
static float    window_sum;       /* running sum of this window's log-mel values */
static uint32_t window_n, frame_idx;
static uint8_t  have_level;

/* The K input columns of the stage being evaluated, in float. Sized by the widest
 * convolution input: channels x frequency bins, times the kernel width. */
#define BG_MAX2(a, b)   ((a) > (b) ? (a) : (b))
#define MAX_CHF         BG_MAX2(BG_CONV0_IN * BG_N_MELS, \
                        BG_MAX2(BG_CONV1_IN * (BG_N_MELS / 2), BG_CONV2_IN * (BG_N_MELS / 4)))
static float   dq[BG_CONV0_K * MAX_CHF];
static float   acc[BG_N_FEATURES];        /* the global average pool */
static uint32_t acc_n;

static const float *conv_w[BG_N_CONV] = { BG_CONV0_W, BG_CONV1_W, BG_CONV2_W };
static const float *conv_b[BG_N_CONV] = { BG_CONV0_B, BG_CONV1_B, BG_CONV2_B };
static const uint16_t conv_in[BG_N_CONV]  = { BG_CONV0_IN,  BG_CONV1_IN,  BG_CONV2_IN };
static const uint16_t conv_out[BG_N_CONV] = { BG_CONV0_OUT, BG_CONV1_OUT, BG_CONV2_OUT };
static const uint8_t  conv_stride[BG_N_CONV] = { BG_CONV0_STRIDE_T, BG_CONV1_STRIDE_T,
                                                 BG_CONV2_STRIDE_T };

static void stage_reset(stage_t *st, int kind, int idx, int k, int s, int p,
                        int ch, int fq, int n_in, uint8_t *ring, int q_in) {
    st->kind = (uint8_t)kind; st->idx = (uint8_t)idx;
    st->k = (uint8_t)k; st->s = (uint8_t)s; st->p = (uint8_t)p;
    st->ch = (uint16_t)ch; st->fq = (uint16_t)fq;
    st->n_in = (uint16_t)n_in;
    st->n_out = (uint16_t)((n_in + 2 * p - k) / s + 1);
    st->seen = 0; st->next_out = 0;
    st->ring = ring; st->q_in = q_in;
    /* The leading `p` columns are zero — as values, so they are the code that
     * dequantises to zero, not the code zero. */
    const uint8_t z = quant(0.0f, q_in);
    memset(ring, z, (size_t)k * ch * fq);
    st->filled = (uint16_t)p;
}

static uint8_t *col_of(stage_t *st, int abs_idx) {
    return st->ring + (size_t)(abs_idx % st->k) * st->ch * st->fq;
}

static void push_col(stage_t *st, const uint8_t *col) {
    memcpy(col_of(st, st->filled), col, (size_t)st->ch * st->fq);
    st->filled++;
    st->seen++;
}

static void push_zero(stage_t *st) {
    const uint8_t z = quant(0.0f, st->q_in);
    memset(col_of(st, st->filled), z, (size_t)st->ch * st->fq);
    st->filled++;
}

static void emit(int s_idx, const uint8_t *col);

/* One output column of a convolution stage, frequency padding 1, then ReLU.
 *
 * The K input columns are dequantised ONCE into `dq`, before the output-channel loop.
 * Reading them straight out of the ring instead costs a dequantisation per
 * multiply-accumulate — the same bytes converted again for every output channel — and
 * the generated code was a hardware divide for the ring index, a byte load, an
 * integer-to-float convert, a subtract and two multiplies, in a serial dependency
 * chain with no fused multiply-add, about 940,000 times per window. With the values
 * already in float the inner loop is a load and an FMA, and the ring arithmetic
 * happens K times per column rather than once per MAC.
 */
static void conv_column(stage_t *st, int a, uint8_t *out) {
    const int ci = st->idx;
    const int ICH = conv_in[ci], OCH = conv_out[ci], K = st->k, F = st->fq;
    const int chF = ICH * F;
    const float *W = conv_w[ci], *B = conv_b[ci];
    const float sc = BG_ACT_SCALE[st->q_in], off = BG_ACT_OFFSET[st->q_in];

    for (int kw = 0; kw < K; kw++) {
        const uint8_t *c = col_of(st, a + kw);
        float *d = &dq[(size_t)kw * chF];
        for (int i = 0; i < chF; i++) d[i] = ((float)c[i] - off) * sc;
    }

    for (int oc = 0; oc < OCH; oc++) {
        const float *w0 = &W[(size_t)oc * ICH * K * K];
        for (int f = 0; f < F; f++) {
            float sum = B[oc];
            const float *w = w0;
            for (int ic = 0; ic < ICH; ic++) {
                for (int kh = 0; kh < K; kh++) {
                    const int fi = f + kh - 1;                 /* SAME padding in freq */
                    if (fi < 0 || fi >= F) { w += K; continue; }
                    const float *v = &dq[(size_t)ic * F + fi];
                    /* K is 3 and the loop body was three instructions, so the compare,
                     * branch and increment around it cost as much as the work. Unrolled
                     * in the same order, so the sum is accumulated identically. */
                    sum += w[0] * v[0];
                    sum += w[1] * v[chF];
                    sum += w[2] * v[2 * chF];
                    w += K;
                }
            }
            /* conv -> quantise -> ReLU -> quantise, the order QuantGate applies. */
            float vq = deq(quant(sum, Q_CONV(ci)), Q_CONV(ci));
            if (vq < 0.0f) vq = 0.0f;
            out[(size_t)oc * F + f] = quant(vq, Q_RELU(ci));
        }
    }
}

/* One output column of a 2x2 max-pool: halves frequency, and the two input columns
 * are the pooling window in time. */
static void pool_column(stage_t *st, int a, uint8_t *out) {
    const int C = st->ch, F = st->fq, FO = F / 2;
    const uint8_t *c0 = col_of(st, a), *c1 = col_of(st, a + 1);
    for (int c = 0; c < C; c++) {
        for (int f = 0; f < FO; f++) {
            uint8_t m = c0[(size_t)c * F + 2 * f];
            const uint8_t cands[3] = { c0[(size_t)c * F + 2 * f + 1],
                                       c1[(size_t)c * F + 2 * f],
                                       c1[(size_t)c * F + 2 * f + 1] };
            for (int j = 0; j < 3; j++) if (cands[j] > m) m = cands[j];
            /* Codes are monotone in value, so the max code IS the max value. It is
             * then requantised onto the pool's own grid. */
            out[(size_t)c * FO + f] = quant(deq(m, st->q_in), Q_POOL(st->idx));
        }
    }
}

static void drain(int s_idx) {
    stage_t *st = &stages[s_idx];
    /* One buffer for every recursion level, which is safe for one reason worth
     * stating: `emit` copies the column into the next stage's ring BEFORE it
     * recurses, and nothing reads this buffer after `emit` returns. */
    static uint8_t out[MAX_COL];
    while (st->next_out < st->n_out &&
           (int)(st->next_out * st->s + st->k) <= (int)st->filled) {
        const int a = st->next_out * st->s;
        if (st->kind == 0) conv_column(st, a, out);
        else               pool_column(st, a, out);
        st->next_out++;
        emit(s_idx, out);
    }
}

/* Hand a finished column to the next stage, or to the global average pool. */
static void emit(int s_idx, const uint8_t *col) {
    if (s_idx + 1 < N_STAGE) {
        stage_t *nx = &stages[s_idx + 1];
        push_col(nx, col);
        drain(s_idx + 1);
        if (nx->seen == nx->n_in) {
            for (int j = 0; j < nx->p; j++) push_zero(nx);
            drain(s_idx + 1);
        }
        return;
    }
    /* Last stage: accumulate for AdaptiveAvgPool2d(1) over frequency and time. */
    const stage_t *st = &stages[s_idx];
    const int C = st->ch, FO = st->fq / 2;
    for (int c = 0; c < C; c++)
        for (int f = 0; f < FO; f++)
            acc[c] += deq(col[(size_t)c * FO + f], Q_POOL(st->idx));
    acc_n += (uint32_t)FO;
}

void bg_fe_reset(void) {
    have_level = 0;
    level = 0.0f;
}

void bg_fe_begin(void) {
    int n = BG_FRAMES_PER_WINDOW;
    uint8_t *rings[N_STAGE] = { ring0, ring1, ring2, ring3, ring4, ring5 };
    int fq = BG_N_MELS;
    for (int i = 0; i < BG_N_CONV; i++) {
        const int q_conv_in = (i == 0) ? Q_IN : Q_POOL(i - 1);
        stage_reset(&stages[2 * i], 0, i, 3, conv_stride[i], 1,
                    conv_in[i], fq, n, rings[2 * i], q_conv_in);
        n = stages[2 * i].n_out;
        stage_reset(&stages[2 * i + 1], 1, i, 2, 2, 0,
                    conv_out[i], fq, n, rings[2 * i + 1], Q_RELU(i));
        n = stages[2 * i + 1].n_out;
        fq /= 2;
    }
    memset(acc, 0, sizeof(acc));
    acc_n = 0;
    window_sum = 0.0f;
    window_n = 0;
    frame_idx = 0;
}

/* Called once the window is complete, so the level it produces applies to the next. */
static void bg_level_update(void) {
    if (window_n) level += BG_IN_LEVEL_ALPHA * (window_sum / (float)window_n - level);
}

float *bg_fe_frame(void) { return bg_logmel_frame(); }

void bg_fe_frame_only(void) {
    float db[BG_N_MELS];
    bg_logmel_column(bg_logmel_frame(), db);
}

void bg_fe_push(void) {
    float db[BG_N_MELS];
    static uint8_t col[BG_N_MELS];
    bg_logmel_column(bg_logmel_frame(), db);

    float fsum = 0.0f;
    for (int m = 0; m < BG_N_MELS; m++) fsum += db[m];

    if (frame_idx == 0) {
        if (!have_level) { level = fsum / (float)BG_N_MELS; have_level = 1; }
        window_centre = level;        /* the state from BEFORE this window */
    }
    window_sum += fsum;
    window_n += BG_N_MELS;
    frame_idx++;

    /* BG_IN_MEAN cancels: the reference adds it and the normaliser removes it. */
    for (int m = 0; m < BG_N_MELS; m++)
        col[m] = quant((db[m] - window_centre) / BG_IN_STD, Q_IN);
    push_col(&stages[0], col);
    drain(0);
    if (stages[0].seen == stages[0].n_in) {
        for (int j = 0; j < stages[0].p; j++) push_zero(&stages[0]);
        drain(0);
    }
}

void bg_fe_end(float *feat) {
    bg_level_update();
    const float inv = acc_n ? 1.0f / (float)acc_n : 0.0f;
    for (int c = 0; c < BG_N_FEATURES; c++)
        feat[c] = deq(quant(acc[c] * inv, Q_AVG), Q_AVG);   /* avgpool -> quantise */
}

float bg_score(const float *feat) {
    float s = BG_LIN_B;
    for (int c = 0; c < BG_N_FEATURES; c++) s += BG_LIN_W[c] * feat[c];
    return s;                                               /* no output quantiser */
}
