# Data attribution

The labels in this folder, and everything in `../demo/`, come from or are derived
from:

> Kahl, S., Wood, C. M., Chaon, P., Peery, M. Z., & Klinck, H. (2022).
> *A collection of fully-annotated soundscape recordings from the Western United
> States* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7050014

recorded in the Lassen and Plumas National Forests, Sierra Nevada, California, and
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
dataset is also one of the evaluation soundscape sets of the BirdSet benchmark
(Rauch et al., 2024).

| file | relationship to the original |
|---|---|
| `annotations.csv`, `species.csv`, `recording_location.txt` | unchanged copies |
| `recordings.csv` | derived: filename, sample rate and duration, read from the FLAC headers |
| `../demo/scores.npz` | derived: per-second model scores computed from the audio |
| `../demo/clips/` | derived: short excerpts trimmed from the original recordings |

The full 33 hours of audio (3.2 GB) are not redistributed here. Download
`soundscape_data.zip` from the Zenodo record and point `data/audio` at the
unzipped folder to rebuild features from scratch.
