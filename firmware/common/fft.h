/* fft.h — in-place radix-2 complex FFT over the tables in model_params.h.
 *
 * Self-contained on purpose. AudioMoth-Project vendors the CMSIS *headers*
 * (arm_math.h declares arm_rfft_fast_f32) but no DSP library, and the Makefile
 * links only -lm -lgcc -lc -lnosys, so calling it would fail at link. Writing our
 * own also means the SAME code runs on host and device, which is what makes
 * test_parity an exact check rather than an approximate one.
 */
#ifndef BIRDGATE_FFT_H
#define BIRDGATE_FFT_H

/* In-place DFT of `n` complex points (n a power of two, n <= BG_FFT_SIZE).
 * Decimation-in-time, so the input is bit-reversed first and output is in order. */
void bg_fft(float *re, float *im, int n);

/* One windowed frame -> power spectrum. `frame` holds BG_FFT_SIZE samples and is
 * DESTROYED (it becomes the FFT's own scratch). Returns BG_N_BINS power bins, in
 * storage this function owns, valid until the next call.
 *
 * Shared by every arm that starts with an FFT, so the Hann window, the real-input
 * packing and the bin convention exist once. An arm that computed its own would be
 * a second thing to keep bit-identical to Python.
 */
const float *bg_power_spectrum(float *frame);

/* W^k = exp(-2 pi i k / BG_FFT_SIZE), k in [0, BG_FFT_SIZE/2). Read through here rather
 * than from the header's tables directly: model_params.h declares them `static`, so
 * every translation unit that touches them links its own 4 KB copy. */
void bg_twiddle(int k, float *wr, float *wi);

#endif /* BIRDGATE_FFT_H */
