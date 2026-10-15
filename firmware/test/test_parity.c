/* test_parity.c — host harness for the whole chain. Raw float32 audio in; per 1 s
 * window out: BG_N_FEATURES features, raw score, the per-window z, the max-pooled z, the
 * gate flag (0/1), and the pre-roll backfill count on an open transition — all float32,
 * BG_N_FEATURES + 5 per window. The recorded mask is gate flag OR backfilled by a later open, exactly as
 * gate.py's `rec[max(0, i - pre):i] = True`.
 *
 *   cc -O2 -Icommon -Iarms/mfcc-dev test/test_parity.c common/fft.c common/scorer.c \
 *      common/adapt.c common/gate.c common/birdgate.c common/bg_stream.c \
 *      arms/mfcc-dev/mfcc.c -lm -o parity
 *   ./parity in.f32 out.f32 [--i16] [--freeze N]
 *
 * `tests/test_c_parity.py` drives this: the same audio through here and through the
 * Python reference, features compared to tolerance and the gate mask compared
 * exactly. If it fails, everything measured in Python is a claim about a machine
 * that does not exist.
 *
 * `--i16` quantises the input to int16 and feeds it through bg_stream.c's ring path
 * instead — the exact code the device runs — using one big ring.
 *
 * Deliberately file-in/file-out rather than linked against Python: the binary built
 * here is compiled from the SAME sources the firmware build uses, with no bindings
 * layer in between that could paper over a difference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "birdgate.h"
#include "arm.h"
#include "adapt.h"

static void emit(FILE *fo, const bg_result_t *r) {
    fwrite(r->feat, sizeof(float), BG_N_FEATURES, fo);
    const float tail[5] = { r->score, r->z_inst, r->z, r->open ? 1.0f : 0.0f,
                            (float)r->backfill };
    fwrite(tail, sizeof(float), 5, fo);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.f32 out.f32 [--i16]\n", argv[0]); return 2; }
    int i16 = 0;
    uint32_t freeze = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--i16") == 0) i16 = 1;
        else if (strcmp(argv[i], "--freeze") == 0 && i + 1 < argc) freeze = (uint32_t)atoi(argv[++i]);
    }

    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror("open input"); return 1; }
    fseek(fi, 0, SEEK_END);
    const long n = ftell(fi) / (long)sizeof(float);
    fseek(fi, 0, SEEK_SET);
    float *x = malloc((size_t)n * sizeof(float));
    if (!x || fread(x, sizeof(float), (size_t)n, fi) != (size_t)n) { perror("read"); return 1; }
    fclose(fi);

    FILE *fo = fopen(argv[2], "wb");
    if (!fo) { perror("open output"); return 1; }

    const long n_win = (n >= BG_WIN_SAMPLES) ? n / BG_WIN_SAMPLES : 0;
    bg_result_t r;
    long fired = 0;

    if (i16) {
        int16_t *q = malloc((size_t)n * sizeof(int16_t));
        for (long i = 0; i < n; i++) {
            float v = x[i] * 32768.0f;
            if (v > 32767.0f) v = 32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            q[i] = (int16_t)v;
        }
        bg_stream_init(q, (uint32_t)n);
        bg_adapt_set_freeze(freeze);
        /* Arrive in AudioMoth-sized slots (16384 samples) to exercise the wrap logic. */
        for (long done = 0; done < n; done += 16384) {
            const long got = (n - done < 16384) ? n - done : 16384;
            bg_stream_set_captured((uint32_t)(done + got));
            while (bg_stream_advance(&r)) { emit(fo, &r); fired += r.open; }
        }
        free(q);
    } else {
        bg_init();
        bg_adapt_set_freeze(freeze);
        for (long w = 0; w < n_win; w++) {
            bg_fe_begin();
            for (int f = 0; f < BG_FRAMES_PER_WINDOW; f++) {
                memcpy(bg_fe_frame(), &x[w * BG_WIN_SAMPLES + f * BG_HOP_SAMPLES],
                       BG_FFT_SIZE * sizeof(float));
                bg_fe_push();
            }
            bg_finish_window(&r);
            emit(fo, &r); fired += r.open;
        }
    }
    fclose(fo);
    free(x);
    fprintf(stderr, "%ld windows x %d values, %ld open\n", n_win, BG_N_FEATURES + 5, fired);
    return 0;
}
