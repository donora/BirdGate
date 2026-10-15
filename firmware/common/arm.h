/* arm.h — the interface every arm provides, and the only thing the shared code knows
 * about the model. `birdgate.c`, `bg_stream.c`, the parity harness and `app/main.c`
 * are written against this, so adding an arm adds files under `arms/<arm>/` and
 * changes nothing else.
 *
 * One decision window, streamed:
 *
 *     bg_fe_begin();
 *     for each of BG_FRAMES_PER_WINDOW frames, hop BG_HOP_SAMPLES:
 *         float *f = bg_fe_frame();   // fill with BG_FFT_SIZE samples in [-1, 1]
 *         bg_fe_push();
 *     bg_fe_end(feat);                // BG_N_FEATURES floats
 *     float s = bg_score(feat);
 *
 * `feat` is whatever vector that arm's scorer consumes, and the split between the two
 * calls is where the arm stops being a feature extractor and starts being a model:
 *
 *   mfcc-dev  40 MFCC window statistics -> a linear model (common/scorer.c)
 *   cnn       32 globally-average-pooled channels -> the final linear layer
 *
 * Both are the input to one dot product, which is why the same two calls fit a linear
 * arm and a convolutional one without a special case anywhere downstream.
 */
#ifndef BIRDGATE_ARM_H
#define BIRDGATE_ARM_H
#include "model_params.h"

/* Once per recording, before any window: clear state that persists ACROSS windows.
 * `bg_fe_begin` runs per window and must not touch it. The CNN's tracked input level
 * is the case in point — carried between windows deliberately, and meaningless
 * carried between recordings, which is also how the Python reference computes it. */
void   bg_fe_reset(void);

void   bg_fe_begin(void);
float *bg_fe_frame(void);
void   bg_fe_push(void);
void   bg_fe_end(float *feat);

/* Provided by common/scorer.c for the linear arms; by the arm itself otherwise. */
float  bg_score(const float *feat);

/* The per-frame front-end work WITHOUT the model: the FFT, the filterbank and the log,
 * and nothing that depends on the arm's classifier. Exists so a self-timing build can
 * separate "the spectrogram is slow" from "the model is slow" in one measurement,
 * which no amount of host timing can settle — a laptop hides exactly the divides and
 * dependency stalls that cost a Cortex-M4. */
void   bg_fe_frame_only(void);

#endif /* BIRDGATE_ARM_H */
