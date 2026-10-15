/* mfcc.h — the mfcc-dev front-end: audio -> the 40 features the scorer consumes.
 *
 * Mirrors `birdgate/frontends.py` for `kind="mfcc-dev"`, the DEVICE-FAITHFUL variant:
 * no centred padding and no global dB clamp, so every value here is one a device can
 * compute from audio it has already heard. See docs/lab-notebook.md §13 G6b.
 *
 * Implements the streaming arm interface in `common/arm.h`; `bg_fe_end` writes
 * BG_N_MFCC means followed by BG_N_MFCC standard deviations, the order
 * `frontends.build` names them in and therefore the order BG_W expects.
 *
 * Memory: one 4 KB frame buffer (which the real-input FFT reuses in place), a 2 KB
 * power spectrum, and 20 running (mean, M2) pairs. No frame ring.
 */
#ifndef BIRDGATE_MFCC_H
#define BIRDGATE_MFCC_H
#include "model_params.h"
#include "arm.h"

#endif /* BIRDGATE_MFCC_H */
