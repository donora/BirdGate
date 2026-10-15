#include "fft.h"
#include "model_params.h"
#include <string.h>

void bg_fft(float *re, float *im, int n) {
    /* Bit-reversal permutation, computed by incrementing a reversed counter rather
     * than reversing each index: j tracks i with its bits mirrored. */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        /* W_len^k == exp(-2*pi*i*k/len), and BG_TW_*[j] == exp(-2*pi*i*j/BG_FFT_SIZE),
         * so the index we want is k * (BG_FFT_SIZE / len). */
        const int step = BG_FFT_SIZE / len;
        const int half = len >> 1;
        /* k outside i, so each twiddle pair is fetched from flash ONCE per stage
         * instead of once per butterfly. The butterflies are independent and write
         * distinct elements, so the reordering is bit-identical — and the tables are
         * indexed with stride `step`, which defeats the flash prefetch, so the saved
         * loads are the expensive kind. */
        for (int k = 0; k < half; k++) {
            const float wr = BG_TW_COS[k * step], wi = BG_TW_SIN[k * step];
            for (int i = 0; i < n; i += len) {
                const float xr = re[i + k + half], xi = im[i + k + half];
                const float vr = xr * wr - xi * wi;
                const float vi = xr * wi + xi * wr;
                const float ur = re[i + k], ui = im[i + k];
                re[i + k] = ur + vr;  im[i + k] = ui + vi;
                re[i + k + half] = ur - vr;  im[i + k + half] = ui - vi;
            }
        }
    }
}

void bg_twiddle(int k, float *wr, float *wi) {
    *wr = BG_TW_COS[k];
    *wi = BG_TW_SIN[k];
}

/* --- the windowed real-input power spectrum, shared by the arms ----------------
 *
 * A complex FFT of BG_FFT_SIZE/2 points on the even/odd split, then the standard
 * real-input recombination. For k in 1..HALF-1,
 *     X[k] = Ze + W^k Zo,   Ze = (Z[k] + conj Z[HALF-k]) / 2,
 *                           Zo = -i (Z[k] - conj Z[HALF-k]) / 2
 * and X[0] = Re Z0 + Im Z0, X[HALF] = Re Z0 - Im Z0. Doing it this way rather than
 * a complex-1024 FFT on real input halves both the work and the buffer.
 */
#define BG_FFT_HALF (BG_FFT_SIZE / 2)

static float bg_pow[BG_N_BINS];

const float *bg_power_spectrum(float *frame) {
    /* The window is symmetric, so walk the two halves separately rather than testing
     * which half each sample is in. Same products, no branch. */
    for (int i = 0; i <= BG_FFT_HALF; i++) frame[i] *= BG_HANN[i];
    for (int i = BG_FFT_HALF + 1; i < BG_FFT_SIZE; i++) frame[i] *= BG_HANN[BG_FFT_SIZE - i];

    float *re = frame, *im = frame + BG_FFT_HALF;
    for (int n = 0; n < BG_FFT_HALF; n++) bg_pow[n] = frame[2 * n + 1];   /* odds out */
    for (int n = 0; n < BG_FFT_HALF; n++) re[n] = frame[2 * n];           /* evens compact */
    memcpy(im, bg_pow, BG_FFT_HALF * sizeof(float));

    bg_fft(re, im, BG_FFT_HALF);

    const float z0r = re[0], z0i = im[0];
    bg_pow[0] = (z0r + z0i) * (z0r + z0i);
    bg_pow[BG_FFT_HALF] = (z0r - z0i) * (z0r - z0i);
    for (int k = 1; k < BG_FFT_HALF; k++) {
        const float ar = re[k], ai = im[k];
        const float br = re[BG_FFT_HALF - k], bi = -im[BG_FFT_HALF - k];   /* conj */
        const float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);
        const float dr = ar - br, di = ai - bi;
        const float or_ = 0.5f * di, oi = -0.5f * dr;                      /* -i/2 (a-b) */
        float wr, wi;
        bg_twiddle(k, &wr, &wi);
        const float xr = er + (or_ * wr - oi * wi);
        const float xi = ei + (or_ * wi + oi * wr);
        bg_pow[k] = xr * xr + xi * xi;
    }
    return bg_pow;
}
