/* gate.h — mirrors birdgate/gate.py::run, step for step. If the two drift, the
 * simulator lies and every number in every report is wrong. */
#ifndef BIRDGATE_GATE_H
#define BIRDGATE_GATE_H
#include <stdint.h>
#include <stdbool.h>
#include "model_params.h"

typedef struct {
    bool    is_open;
    int32_t i;            /* window index within this recording */
    int32_t opened_at;
    int32_t last_pos;
} bg_gate_t;

void bg_gate_init(bg_gate_t *g);

/* One window. `z` is the aggregated z-score. Returns whether THIS window is recorded.
 * On an open transition `*backfill` is set to the number of earlier windows to
 * recover from the ring buffer (the pre-roll rewind); otherwise 0. */
bool bg_gate_step(bg_gate_t *g, float z, int *backfill);

#endif /* BIRDGATE_GATE_H */
