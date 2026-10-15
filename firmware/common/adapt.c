#include "adapt.h"
#include <stdlib.h>
#include <string.h>

static float    ring[BG_ADAPT_N];      /* the previous BG_ADAPT_N raw scores */
static float    sorted[BG_ADAPT_N];    /* sort scratch, one sort per window */
static float    zhist[BG_AGG_N];       /* the last BG_AGG_N z-scores, ring */
static uint32_t head, zhead, n_seen;
static uint32_t freeze_after;          /* 0 = never freeze */

void bg_adapt_init(void) { head = zhead = n_seen = 0; }

void bg_adapt_set_freeze(uint32_t n_windows) { freeze_after = n_windows; }

static int cmp_float(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

/* np.quantile, method="linear", including numpy's two-sided lerp form. */
static float quantile(void) {
    memcpy(sorted, ring, sizeof(sorted));
    qsort(sorted, BG_ADAPT_N, sizeof(float), cmp_float);
    const double pos = (double)BG_ADAPT_QUANTILE * (double)(BG_ADAPT_N - 1);
    const int lo = (int)pos;
    const float t = (float)(pos - (double)lo);
    const float a = sorted[lo];
    const float b = sorted[lo + 1 < BG_ADAPT_N ? lo + 1 : lo];
    const float d = b - a;
    return t >= 0.5f ? b - d * (1.0f - t) : a + d * t;
}

float bg_adapt_step(float score, float *z_instant) {
    if (n_seen == 0) {
        for (int i = 0; i < BG_ADAPT_N; i++) ring[i] = score;   /* adapt.py's leading pad */
    }
    const float bg = quantile();                                 /* strictly past scores */
    if (!freeze_after || n_seen < freeze_after) {
        ring[head] = score;
        head = (head + 1) % BG_ADAPT_N;
    }

    float sc = BG_GLOBAL_SCALE;
    if (sc < BG_MIN_SCALE) sc = BG_MIN_SCALE;
    const float z = (score - bg) / sc;
    if (z_instant) *z_instant = z;

    if (n_seen == 0) {
        for (int i = 0; i < BG_AGG_N; i++) zhist[i] = z;        /* adapt.aggregate's pad */
    }
    zhist[zhead] = z;
    zhead = (zhead + 1) % BG_AGG_N;
    n_seen++;

    float m = zhist[0];
    for (int i = 1; i < BG_AGG_N; i++) if (zhist[i] > m) m = zhist[i];
    return m;
}
