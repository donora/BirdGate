/* logmel.h — the CNN arm's front-end: one frame of audio -> BG_N_MELS log-mel dB.
 *
 * Mirrors `birdgate/frontends.py::logmel(dev=True)`, which is the causal variant:
 * `center=False`, so a frame only ever covers samples already heard, and
 * `top_db=None`, so nothing later rewrites a value already computed.
 *
 * Same FFT, Hann window and mel convention as every other arm — all of it lives in
 * `common/fft.c`. What differs from `mfcc-dev` is only that there are 32 bands
 * instead of 128 and no DCT afterwards, which is why this arm is the cheaper of the
 * two in flash despite carrying a network.
 */
#ifndef BIRDGATE_LOGMEL_H
#define BIRDGATE_LOGMEL_H
#include "model_params.h"

/* The buffer to fill with BG_FFT_SIZE samples in [-1, 1]. Owned here so the arm has
 * exactly one, and it doubles as the FFT's scratch. */
float *bg_logmel_frame(void);

/* `frame` holds BG_FFT_SIZE samples and is destroyed; writes BG_N_MELS dB values. */
void bg_logmel_column(float *frame, float *db);

#endif /* BIRDGATE_LOGMEL_H */
