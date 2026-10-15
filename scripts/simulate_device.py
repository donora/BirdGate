"""Run the whole gate over a real 1-hour recording, bit-honestly.

Reports what a real SD card would contain (bytes, regions, duty cycle) and — the
part that matters — what was LOST: every annotation box dropped or clipped,
sorted by species, so the failure mode is legible rather than a percentage.

Also writes a before/after audio pair. That demo convinces people; tables do not.
"""
