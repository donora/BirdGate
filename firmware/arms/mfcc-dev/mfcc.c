#include "mfcc.h"
#include "fft.h"
#include <math.h>
#include <string.h>

static float    frame[BG_FFT_SIZE];   /* the caller's samples; FFT scratch thereafter */
static float    mean[BG_N_MFCC], m2[BG_N_MFCC];
static uint32_t n_frames;

void bg_fe_reset(void) { }   /* nothing persists across windows in this arm */

void bg_fe_begin(void) {
    memset(mean, 0, sizeof(mean));
    memset(m2, 0, sizeof(m2));
    n_frames = 0;
}

float *bg_fe_frame(void) { return frame; }

void bg_fe_frame_only(void) {
    /* Everything bg_fe_push does except the running mean and variance, which is the
     * only part that belongs to the model rather than the spectrogram. */
    const float *pow_ = bg_power_spectrum(frame);
    float db[BG_N_MELS];
    for (int m = 0; m < BG_N_MELS; m++) {
        const int s = BG_MEL_START[m], n = BG_MEL_LEN[m], o = BG_MEL_OFF[m];
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += BG_MEL_W[o + j] * pow_[s + j];
        db[m] = 10.0f * log10f(acc > BG_LOG_FLOOR ? acc : BG_LOG_FLOOR);
    }
    volatile float sink = 0.0f;
    for (int k = 0; k < BG_N_MFCC; k++) {
        const float *row = &BG_DCT[k * BG_N_MELS];
        float acc = 0.0f;
        for (int m = 0; m < BG_N_MELS; m++) acc += row[m] * db[m];
        sink += acc;
    }
    (void)sink;
}

void bg_fe_push(void) {
    const float *pow_ = bg_power_spectrum(frame);

    float db[BG_N_MELS];
    for (int m = 0; m < BG_N_MELS; m++) {
        const int s = BG_MEL_START[m], n = BG_MEL_LEN[m], o = BG_MEL_OFF[m];
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += BG_MEL_W[o + j] * pow_[s + j];
        /* power_to_db with ref=1.0: clamp at amin BEFORE the log, never after. */
        db[m] = 10.0f * log10f(acc > BG_LOG_FLOOR ? acc : BG_LOG_FLOOR);
    }

    /* DCT-II, then Welford's running mean / M2 per coefficient. Exact up to float
     * summation order against numpy's two-pass mean and std over the window. */
    n_frames++;
    for (int k = 0; k < BG_N_MFCC; k++) {
        const float *row = &BG_DCT[k * BG_N_MELS];
        float c = 0.0f;
        for (int m = 0; m < BG_N_MELS; m++) c += row[m] * db[m];
        const float d = c - mean[k];
        mean[k] += d / (float)n_frames;
        m2[k]   += d * (c - mean[k]);
    }
}

void bg_fe_end(float *feat) {
    for (int k = 0; k < BG_N_MFCC; k++) {
        feat[k] = mean[k];                                                 /* mfccNN_mean */
        feat[BG_N_MFCC + k] = n_frames ? sqrtf(m2[k] / (float)n_frames) : 0.0f; /* mfccNN_std */
    }
}
