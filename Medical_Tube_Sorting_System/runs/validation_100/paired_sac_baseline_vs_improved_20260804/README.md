# Paired YOLO and SAC validation

The two visible PyBullet tests used the same simulation seed, the same 100 tube classes, the same lateral offsets, the same tube parameters, and the same improved SAC policy. The comparison isolates the effect of the full-system control interface.

The initial interface used a valve deadband of 0.60 for all five jets. The calibrated interface used deadbands of 0.85, 0.85, 0.60, 0.70, and 0.85. A command below the relevant value was not converted into a pneumatic pulse. The controller waited for the next 30 Hz update. The SAC action remained continuous and no tube mass, conveyor speed, jet reference impulse, or geometry was changed.

## Main results

| Configuration | Correct | Recognition failures | Nozzle selection failures | Physical failures | Success rate |
|---|---:|---:|---:|---:|---:|
| Initial interface | 81 | 4 | 0 | 15 | 81% |
| Calibrated interface | 90 | 4 | 0 | 6 | 90% |

The calibrated interface fixed nine paired trials and caused no regression. The physical failure count decreased from 15 to 6. Recognition performance did not change because the YOLO model and camera settings were unchanged.

## Interpretation

This comparison is not a second SAC training result. It is an actuator-interface calibration using the already trained continuous SAC policy. The separate five-seed SAC comparison is stored in `runs/rl/comparisons/sac_improved_v1` and evaluates the reward-based policy improvement.

The remaining end-to-end failures were concentrated in the universal polypropylene and lysis tube classes. These tubes reached the correct jet, but some did not remain below the collection-bin rim. The vision model also produced four recognition failures.

## Files

- `end_to_end_comparison.csv`: main result table.
- `end_to_end_comparison.json`: complete summary and paired transitions.
- `overall_and_failures.png`: overall success and failure decomposition.
- `class_success_rate.png`: result by tube class.
- `paired_outcome_changes.png`: paired changes for the 100 common tubes.
- `position_impulse_outcomes.png`: physical outcome against estimated lateral position and effective impulse.
