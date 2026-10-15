#include "gate.h"
#include <math.h>

void bg_gate_init(bg_gate_t *g) {
    g->is_open = false;
    g->i = 0;
    g->opened_at = g->last_pos = -1;
}

bool bg_gate_step(bg_gate_t *g, float z, int *backfill) {
    const int pre = (int)ceilf(BG_PRE_ROLL_S / BG_WIN_S);   /* ceil, as gate.py */
    int recheck = (int)(BG_RECHECK_S / BG_WIN_S + 0.5f);
    if (recheck < 1) recheck = 1;
    const int32_t i = g->i;
    *backfill = 0;

    if (g->is_open) {
        const int32_t elapsed = i - g->opened_at;
        /* The classifier is only awake on recheck instants. */
        if (elapsed > 0 && elapsed % recheck == 0) {
            if (z > BG_THETA_OFF) g->last_pos = i;
            const float quiet_for = (float)(i - g->last_pos) * BG_WIN_S;
            if (quiet_for > BG_HANGOVER_S && (float)elapsed * BG_WIN_S >= BG_MIN_ON_S) {
                g->is_open = false;
            }
        }
    }
    if (!g->is_open && z > BG_THETA_ON) {
        g->is_open = true;
        g->opened_at = g->last_pos = i;
        *backfill = pre < i ? pre : i;            /* rec[max(0, i - pre):i] */
    }
    g->i = i + 1;
    return g->is_open;
}
