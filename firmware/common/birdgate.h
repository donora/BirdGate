/* birdgate.h — the whole detection chain behind two calls, so the host harness and the
 * device glue cannot run different pipelines.
 *
 *   bg_init();
 *   ... for each 1 s window: bg_fe_begin / frame / push x BG_FRAMES_PER_WINDOW ...
 *   bg_finish_window(&r);          // -> features, score, z, gate state
 *
 * bg_stream.c wraps this for the device's int16 external-SRAM ring.
 */
#ifndef BIRDGATE_H
#define BIRDGATE_H
#include <stdint.h>
#include <stdbool.h>
#include "model_params.h"
#include "adapt.h"   /* bg_adapt_set_freeze: the bench knob app/main.c reaches for */
#include "arm.h"     /* the arm interface, so app/main.c can drive it directly */

typedef struct {
    float feat[BG_N_FEATURES];
    float score;      /* raw margin */
    float z_inst;     /* background-relative, THIS window only: per-window detection */
    float z;          /* background-relative, max-pooled: what the gate sees */
    bool  open;       /* gate state after this window == this window is recorded */
    int   backfill;   /* on an open transition: earlier windows to recover (pre-roll) */
} bg_result_t;

void bg_init(void);
void bg_finish_window(bg_result_t *r);   /* after the last bg_fe_push of a window */
bool bg_gate_open(void);

/* Device glue: samples live in an int16 ring (the AudioMoth's external SRAM). */
void bg_stream_init(const int16_t *ring, uint32_t ring_samples);
/* Total samples the DMA has captured since bg_stream_init. Must be derived from the
 * capture side (the DMA transfer counter), never from how many buffers this loop has
 * got around to consuming: if the detector is slower than real time those two diverge,
 * and a read position built from the second one drifts quietly into audio that has
 * already been overwritten while every internal check still looks consistent. */
void bg_stream_set_captured(uint32_t total_samples);
/* Process the next whole window if enough samples have arrived. Call until false. */
bool bg_stream_advance(bg_result_t *r);

/* Windows skipped because the DMA lapped the ring while we were computing. Any value
 * but zero means the detector is not keeping up with real time, and the figure is the
 * one to look at before trusting anything it reported. */
uint32_t bg_stream_dropped(void);

#endif /* BIRDGATE_H */
