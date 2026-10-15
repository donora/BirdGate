#include "logmel.h"
#include "fft.h"
#include <math.h>

static float frame_buf[BG_FFT_SIZE];

float *bg_logmel_frame(void) { return frame_buf; }

void bg_logmel_column(float *frame, float *db) {
    const float *pow_ = bg_power_spectrum(frame);
    for (int m = 0; m < BG_N_MELS; m++) {
        const int s = BG_MEL_START[m], n = BG_MEL_LEN[m], o = BG_MEL_OFF[m];
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += BG_MEL_W[o + j] * pow_[s + j];
        /* power_to_db with ref=1.0: clamp at amin BEFORE the log, never after. */
        db[m] = 10.0f * log10f(acc > BG_LOG_FLOOR ? acc : BG_LOG_FLOOR);
    }
}
