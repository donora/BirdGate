/* scorer.c — mirrors birdgate/svm.py: one dot product and a bias.
 *
 * The StandardScaler is already folded into BG_W / BG_BIAS (svm.fold), so the device
 * applies them to raw features directly.
 *
 * Lives under the arm rather than in `common/` for a build reason, not a design one:
 * `build.sh` copies all of `common/`, and the CNN arm defines its own `bg_score`, so a
 * shared copy would be a duplicate symbol at link. If a second linear arm is ever
 * ported, promote this back and teach `build.sh` which shared files each arm wants.
 */
#include "arm.h"

float bg_score(const float *feat) {
    float acc = 0.0f;
    for (int i = 0; i < BG_N_FEATURES; i++) acc += BG_W[i] * feat[i];
    return acc + BG_BIAS;
}
