# Gate versus cross-attention fusion

Matched E3/E4/E5 training runs, seeds 42/52/62, and fixed validation views.
Positive delta means cross-attention is higher. Lower MAE is better.

| Experiment | Split and metric | Gate mean +/- SD | Cross mean +/- SD | Cross - gate |
|---|---|---:|---:|---:|
| E3 | valid clean_macro_f1 | 0.5330 +/- 0.0108 | 0.5215 +/- 0.0093 | -0.0115 |
| E3 | valid primary_macro_f1_missing36 | 0.5199 +/- 0.0118 | 0.5078 +/- 0.0166 | -0.0121 |
| E3 | valid clean_mae | 0.7167 +/- 0.0127 | 0.7221 +/- 0.0388 | +0.0054 |
| E3 | valid single_missing_mae | 0.7290 +/- 0.0125 | 0.7329 +/- 0.0389 | +0.0039 |
| E3 | test clean_macro_f1 | 0.5174 +/- 0.0121 | 0.5033 +/- 0.0183 | -0.0142 |
| E3 | test clean_mae | 0.7897 +/- 0.0117 | 0.8373 +/- 0.0330 | +0.0476 |
| E4 | valid clean_macro_f1 | 0.5334 +/- 0.0176 | 0.5263 +/- 0.0064 | -0.0071 |
| E4 | valid primary_macro_f1_missing36 | 0.5182 +/- 0.0143 | 0.5097 +/- 0.0128 | -0.0085 |
| E4 | valid clean_mae | 0.7021 +/- 0.0113 | 0.7181 +/- 0.0138 | +0.0159 |
| E4 | valid single_missing_mae | 0.7141 +/- 0.0103 | 0.7323 +/- 0.0143 | +0.0182 |
| E4 | test clean_macro_f1 | 0.5234 +/- 0.0097 | 0.5281 +/- 0.0238 | +0.0047 |
| E4 | test clean_mae | 0.7889 +/- 0.0094 | 0.8063 +/- 0.0266 | +0.0174 |
| E5 | valid clean_macro_f1 | 0.5733 +/- 0.0044 | 0.5416 +/- 0.0093 | -0.0317 |
| E5 | valid primary_macro_f1_missing36 | 0.5583 +/- 0.0005 | 0.5256 +/- 0.0097 | -0.0328 |
| E5 | valid clean_mae | 0.6861 +/- 0.0275 | 0.7179 +/- 0.0128 | +0.0318 |
| E5 | valid single_missing_mae | 0.6990 +/- 0.0244 | 0.7377 +/- 0.0155 | +0.0387 |
| E5 | test clean_macro_f1 | 0.5541 +/- 0.0069 | 0.5485 +/- 0.0164 | -0.0057 |
| E5 | test clean_mae | 0.7489 +/- 0.0041 | 0.7861 +/- 0.0167 | +0.0372 |

## Paired validation differences

The interval resamples 728 validation sample IDs; all repeated views and three
fixed seeds for each sample stay in the same cluster. Checkpoints were selected
on this validation set, so these intervals are descriptive rather than an
independent generalization claim.

| Experiment | Subset | Metric | Cross - gate | 95% cluster interval |
|---|---|---|---:|---:|
| E3 | clean | accuracy | -0.0188 | [-0.0435, +0.0046] |
| E3 | clean | mae | +0.0054 | [-0.0197, +0.0291] |
| E3 | missing36 | accuracy | -0.0191 | [-0.0377, -0.0013] |
| E3 | missing36 | mae | +0.0039 | [-0.0164, +0.0248] |
| E4 | clean | accuracy | -0.0220 | [-0.0444, -0.0000] |
| E4 | clean | mae | +0.0159 | [-0.0083, +0.0396] |
| E4 | missing36 | accuracy | -0.0188 | [-0.0355, -0.0024] |
| E4 | missing36 | mae | +0.0182 | [-0.0019, +0.0397] |
| E5 | clean | accuracy | -0.0462 | [-0.0687, -0.0238] |
| E5 | clean | mae | +0.0318 | [+0.0095, +0.0526] |
| E5 | missing36 | accuracy | -0.0449 | [-0.0616, -0.0277] |
| E5 | missing36 | mae | +0.0387 | [+0.0201, +0.0574] |

## Single-modality missing conditions

Values are macro-F1 changes averaged across three seeds and 12 conditions
per modality. T70 contains the three text positions at 70% missing.

| Experiment | T | A | V | T70 |
|---|---:|---:|---:|---:|
| E3 | -0.0093 (13/36 wins) | -0.0132 (11/36 wins) | -0.0139 (12/36 wins) | -0.0120 (3/9 wins) |
| E4 | -0.0135 (9/36 wins) | -0.0013 (15/36 wins) | -0.0107 (12/36 wins) | -0.0213 (2/9 wins) |
| E5 | -0.0344 (2/36 wins) | -0.0334 (0/36 wins) | -0.0304 (0/36 wins) | -0.0368 (1/9 wins) |

## Conclusion

The new fusion does not improve the primary 36-condition validation macro-F1.
E5 falls for every seed and its test MAE also rises. E4 has a small test
macro-F1 gain but falls on validation and worsens test MAE; this is not
enough evidence to replace the gate baseline.

Source files: each run's config.json, run_info.json, eval_valid.json,
eval_valid_samples.npz and eval_test.json. The legacy seed-52/62 test
results are under legacy_gate_recheck/; historical artifacts were not overwritten.
