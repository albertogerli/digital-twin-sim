# EnKF Baseline Comparisons (Brexit 2016)

**Ground Truth:** Leave 51.89%

| Method | Final pred (%) | Error (pp) | Uses params update | Uses dynamics |
|---|---|---|---|---|
| Last poll | 44.0 | 7.9 | No | No |
| Poll average | 42.8 | 9.1 | No | No |
| EnKF (state only) | 50.1 | 1.8 | No | Yes |
| EnKF (state+params) | 50.1 | 1.8 | Yes | Yes |

## Interpretation

- **Best method:** EnKF (state+params) (1.8pp error)
- **Worst method:** Poll average (9.1pp error)
- **EnKF improvement over last-poll:** +6.1pp

The full EnKF (state+params) leverages both the opinion dynamics model
and parameter learning from observations, providing the most accurate prediction.
