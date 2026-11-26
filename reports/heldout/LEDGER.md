# Held-Out Evaluation Ledger

Append-only audit log of official held-out evaluations.

**Rules (R1, effective 2026-11-15):**
1. Held-out v1 (`100010..100089`) is retired into `benchmarks/suites/heldout_v1_retired.yaml` (`split: dev`) after 8 runs during early Phase 2.
2. Held-out v2 (`benchmarks/suites/heldout.yaml`, seeds `200010..200129`, including unseen `partial_observability` and `stochastic_dynamic` families) may ONLY be executed on a clean `HEAD` carrying an explicit git tag `eval-*` (cadence: once per milestone).
3. `EvaluationProtocol` enforces the `eval-*` tag gate and automatically appends a row below on completion.

## Retired Held-Out v1 History (n = 1 seed/task, pre-R1)

| Timestamp (UTC) | Git SHA | Tag | Suite | Mode | Zero-Shot | Replay (5-ep) | Report |
|---|---|---|---|---|---|---|---|
| 20261110_024939 | `aa0bca54` | *(pre-gate)* | v1 (retired) | handcrafted | 0.0000 | 0.0000 | `aa0bca54a84f8f491b4bee309406ae0ac4b42201_handcrafted_20261110_024939.json` |
| 20261110_042821 | `bd076556` | *(pre-gate)* | v1 (retired) | handcrafted | 0.0000 | 0.0000 | `bd076556eccb325cf11c91adff2bf9edd1c51f21_handcrafted_20261110_042821.json` |
| 20261110_142352 | `1ba1f412` | *(pre-gate)* | v1 (retired) | handcrafted | 0.2100 | 0.2067 | `1ba1f412…_handcrafted_20261110_142352.json` |
| 20261110_215317 | `1c32d02f` | *(pre-gate)* | v1 (retired) | handcrafted | 0.4639 | 0.3133 | `1c32d02f68a618939120d8795b0d917c92479566_handcrafted_20261110_215317.json` |
| 20261111_103323 | `468d09fc` | *(pre-gate)* | v1 (retired) | handcrafted | 0.4622 | 0.5242 | `468d09fc03ca6369128430d357bc7b1487a9d6d1_handcrafted_20261111_103323.json` |
| 20261113_081615 | `aeca4f6e` | *(pre-gate)* | v1 (retired) | handcrafted | 0.6652 | 0.8744 | `aeca4f6e0a0f96663643ccdbe7d623e44cbb7035_handcrafted_20261113_081615.json` |
| 20261113_093702 | `edbc9c41` | *(pre-gate)* | v1 (retired) | handcrafted | 0.7335 | 0.8753 | `edbc9c413ff7a8cb76d9e0e9a54bbe29745ef511_handcrafted_20261113_093702.json` |
| 20261114_065659 | `562e1c82` | *(pre-gate)* | v1 (retired) | handcrafted | 0.8339 | 0.8744 | `562e1c82ffaebe2e83d3a8767718b85e5d1d35d8_handcrafted_20261114_065659.json` |

## Sealed Held-Out v2 Runs (all-seed n ≥ 20/task, Wilson 95% CI)

| Timestamp (UTC) | Git SHA | Eval Tag | Config Hash | Env Hash | Mode | Zero-Shot (Solve Rate, 95% CI) | Cross-Seed Transfer (95% CI) | Report |
|---|---|---|---|---|---|---|---|---|
| 20261117_054729 | `f0587fc1` | `eval-phase2-r9` | `b14d80e89bc8cc3e` | `fc495a2dba8a1a49` | handcrafted | 0.6472 (sr=0.6083, CI=[0.5189, 0.691]) | 0.6930 (CI=[0.6032, 0.7702]) | `f0587fc183b07a732fb7886d4b4bb3473f79c730_handcrafted_20261117_054729.json` |
| 20261121_070917 | `a26f09bd` | `eval-phase2-r10` | `ab50c10ca4881dae` | `aec58d7269d2cd6d` | handcrafted | 0.8236 (sr=0.9250, CI=[0.8636, 0.96]) | 0.9035 (CI=[0.8354, 0.9453]) | `a26f09bd4605752d9b2a3cc5c298b29c634dd420_handcrafted_20261121_070917.json` |
| 20261126_112547 | `072b4cb1` | `eval-phase3-final` | `ab50c10ca4881dae` | `aec58d7269d2cd6d` | handcrafted | 0.8359 (sr=0.9417, CI=[0.8845, 0.9715]) | 0.9035 (CI=[0.8354, 0.9453]) | `072b4cb1cc429d9959fc48abe879b8e3be1432d0_handcrafted_20261126_112547.json` |
