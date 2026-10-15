#include "birdgate.h"
#include "arm.h"
#include "adapt.h"
#include "gate.h"

static bg_gate_t gate;

void bg_init(void) {
    bg_adapt_init();
    bg_gate_init(&gate);
    bg_fe_reset();
    bg_fe_begin();
}

void bg_finish_window(bg_result_t *r) {
    bg_fe_end(r->feat);
    r->score = bg_score(r->feat);
    r->z = bg_adapt_step(r->score, &r->z_inst);
    r->open = bg_gate_step(&gate, r->z, &r->backfill);
    bg_fe_begin();
}

bool bg_gate_open(void) { return gate.is_open; }
