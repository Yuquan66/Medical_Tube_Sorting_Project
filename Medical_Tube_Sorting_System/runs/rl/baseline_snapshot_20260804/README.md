# Formal RL baseline snapshot

Created on 2026-08-04 before the next SAC reward experiment.

The snapshot contains the four formal seed 42 policies, their training and
evaluation records, the five-seed common comparison, and the source files used
to reproduce the baseline behaviour.

The shared environment source includes configurable reward parameters. Its
default values preserve the original baseline reward:

- `energy_penalty = 0.02`
- `correct_jet_intensity_weight = 0.50`

The stored policies and result files were not regenerated or overwritten.
`SHA256SUMS.csv` records a checksum for every archived file.
