/* adapt.h — mirrors birdgate/adapt.py for the ONE configuration the device runs:
 * mode="quantile", scale="global", agg="max". export.py refuses to emit anything else.
 *
 * Per window: background = BG_ADAPT_QUANTILE of the previous BG_ADAPT_N raw scores
 * (pre-history filled with the first score, as adapt.normalise pads), then
 * z = (score - background) / max(BG_GLOBAL_SCALE, BG_MIN_SCALE), then the max over
 * the last BG_AGG_N z-scores (pre-history filled with the first z).
 *
 * Never divide by a local spread — adapt.py measured that it destroys
 * within-recording discrimination (AUC 0.925 -> 0.74).
 */
#ifndef BIRDGATE_ADAPT_H
#define BIRDGATE_ADAPT_H
#include "model_params.h"

void  bg_adapt_init(void);

/* Bench aid. After `n_windows` have been seen, stop updating the background and hold
 * it, so a threshold stops moving under repeated testing. 0 (the default, and what
 * ships) keeps tracking for ever, which is the whole point of CFAR in the field. */
void  bg_adapt_set_freeze(uint32_t n_windows);

/* Feed one raw score; returns the aggregated z-score the gate should see.
 * `z_instant`, if non-NULL, receives this window's OWN z before the max-of-N pooling
 * — the per-window "is this second birdsong?" decision variable. The gate wants the
 * pooled value; a live detection indicator wants this one, because pooling keeps a
 * detection alive for BG_AGG_N - 1 further windows. */
float bg_adapt_step(float score, float *z_instant);

#endif /* BIRDGATE_ADAPT_H */
