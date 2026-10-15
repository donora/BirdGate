/* bg_stream.c — frames 1 s windows straight out of the device's int16 sample ring.
 *
 * The ring is the AudioMoth's external SRAM: NUMBER_OF_BUFFERS contiguous slots, so
 * an absolute sample position maps to ring[pos % ring_samples]. Windows are
 * consecutive BG_WIN_SAMPLES blocks from the first sample; frame f of a window starts
 * f * BG_HOP_SAMPLES in and is BG_FFT_SIZE long, exactly as frontends.raw_templates
 * indexes librosa's center=False frames. Samples after the last whole frame are
 * consumed and ignored, as Python ignores them.
 *
 * int16 -> float is x / 32768: Python trains on librosa's float audio in [-1, 1].
 */
#include "birdgate.h"
#include "arm.h"

static const int16_t *ring;
static uint32_t ring_n, avail, next_end, dropped;

void bg_stream_init(const int16_t *r, uint32_t ring_samples) {
    ring = r;
    ring_n = ring_samples;
    avail = 0;
    next_end = BG_WIN_SAMPLES;
    dropped = 0;
    bg_init();
}

void bg_stream_set_captured(uint32_t total_samples) { avail = total_samples; }

static void fetch_frame(uint32_t start, float *dst) {
    uint32_t p = start % ring_n;
    for (int i = 0; i < BG_FFT_SIZE; i++) {
        dst[i] = (float)ring[p] * (1.0f / 32768.0f);
        if (++p == ring_n) p = 0;
    }
}

uint32_t bg_stream_dropped(void) { return dropped; }

bool bg_stream_advance(bg_result_t *r) {
    if (avail < next_end) return false;

    /* If the DMA has lapped the ring past the window we are about to read, that audio
     * is partly overwritten. Skip to the newest complete window instead.
     *
     * Without this, falling behind does not merely make the detector late, it makes it
     * WRONG: it goes on computing features from a buffer being rewritten underneath it,
     * which looks like a model that has stopped recognising anything rather than like a
     * device that is out of time. `bg_stream_dropped` counts the skips, so the two are
     * distinguishable. */
    if (avail - (next_end - BG_WIN_SAMPLES) > ring_n) {
        next_end = avail;
        dropped++;
    }

    const uint32_t start = next_end - BG_WIN_SAMPLES;
    for (int f = 0; f < BG_FRAMES_PER_WINDOW; f++) {
        fetch_frame(start + (uint32_t)f * BG_HOP_SAMPLES, bg_fe_frame());
        bg_fe_push();
    }
    bg_finish_window(r);
    next_end += BG_WIN_SAMPLES;
    return true;
}
