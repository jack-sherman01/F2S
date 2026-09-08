# Reproduction report: the full F2S experimental record, re-run on NHR@FAU Alex

**Date:** 2026-09-08 · **Cluster:** NHR@FAU Alex (A40 / A100) · **Baseline checkpoint:**
`results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-09-04-22-44-00/ckpt/policy_last.ckpt`
· **Total compute:** ~6 h GPU across 9 Slurm jobs

Every experiment in `README_F2S.md` was re-executed from a checkpoint retrained on this
cluster, using the committed analysis scripts with their originally recorded parameters.
All output was written to `*_alex` directories; nothing recorded on the original
`/data/heng/F2S` machine was overwritten, so every number below is a direct comparison.

Two code changes were needed to make the re-run possible at all; both are described in
[§6](#6-reproducibility-gaps-found) and neither alters any algorithm.

---

## 1. Summary

| | |
|---|---|
| **Reproduced** | SOE 0%, Unguided Latent Repair 0%, Failure Replay mean, empty skill archive, 2 validated skills, the capstone recoverability predictor, the contact-dynamics gap, Lift skill rejection, ensemble disagreement |
| **Not reproduced** | F2S's margin over Fixed Policy (**+2.2 points → exactly 0**), the world model's h=16 RMSE threshold claim |
| **New findings** | Evaluation is deterministic run-to-run; the 10-point spread traced to process context, not noise. Five artifacts or code paths in the repository cannot regenerate the results committed beside them. |

The headline: **every negative result in the project reproduces, and the one positive
result does not.**

---

## 2. Main table — in-distribution, 30 episodes per seed

| method | Alex (seed 0 / 1 / 2) | mean | original machine | mean |
|---|---|---|---|---|
| Fixed Policy | 83.3 / 80.0 / 70.0 | **77.8%** | 73.3 / 73.3 / 70.0 | 72.2% |
| **F2S** | **83.3 / 80.0 / 70.0** | **77.8%** | 80.0 / 73.3 / 70.0 | 74.4% |
| Failure Replay | 63.3 / 70.0 / 63.3 | **65.6%** | 63.3 / 66.7 / 66.7 | **65.6%** |
| SOE | 0.0 / 0.0 / 0.0 | 0.0% | 0.0 / 0.0 / 0.0 | 0.0% |
| Unguided Latent Repair | 0.0 / 0.0 / 0.0 | 0.0% | 0.0 / 0.0 / 0.0 | 0.0% |

Unseen configuration (100 episodes, seed 0): Fixed Policy 43% (was 45%), **F2S 43%**
(was 45%), Failure Replay 44% (was 36%), SOE 1% (was 0%).

### 2.1 F2S equals Fixed Policy, seed by seed, because no skill ever fires

`skills_used` is empty in **all 90 F2S evaluation episodes** across the three seeds. The
two archived skills carry a spatial precondition — a specific `object_xy` with a 3 cm
tolerance — and no failure encountered during evaluation fell inside it. With retrieval
never firing, F2S reduces to Fixed Policy by construction, which is exactly what the
per-seed numbers show.

On the original machine the two methods also agree on seeds 1 and 2. The entire reported
margin is **seed 0, where two episodes out of thirty differ**. That is the whole of
"F2S 74.4% vs Fixed Policy 72.2%".

This is the outcome the capstone finding predicts: a skill is only ever validated at a
near-miss state, its applicability region is the 3 cm neighbourhood Day-19 validation
actually tested, and a different checkpoint fails in different places, so the region is
never entered again.

### 2.2 Failure Replay reproduces exactly, and its negative result strengthens

Mean 65.6% on both machines. Against the Alex baseline's 77.8% this is a **12.2-point
regression** from naive replay, against 6.6 points originally — the same conclusion, with
more room in it: behaviour cloning on raw failed trajectories has no reward signal and
simply teaches the policy to imitate its failures a little more.

---

## 3. Evolution loop and skill discovery

### 3.1 Three rounds, empty archive

| | original | Alex (CEM path) |
|---|---|---|
| total wall time | 848.1 s | 857.4 s |
| round 0 success rate | 0.7400 | **0.7400** |
| round 0 mean episode length | 181.54 | 181.36 |
| **final archive size** | **0** | **0** |

On the CEM path the Alex run generated **10,560 candidates and executed 66 for real
across three rounds, archiving none** — consistent with the Day-24 ablation's 0/1278.

The candidate-generation path matters and was initially mismatched: the original
`f2s_final` predates CEM becoming the default and generated 96/160/128 candidates per
round. With `--no_use_cem` restored ([§6.1](#61-a-parameter-the-cli-never-exposed)) the
counts line up: `f2s_dev` **48/32** against 48/32 and `f2s_dev_cem` **960/1600** against
960/1600, exact in both rounds. Round 2 of both dev runs reached `too_few_failures`
(1 and 0 failure segments) where the original still had enough to search; at 10 episodes
per round these counts are single digits.

### 3.2 Per-state offset sweep

| | original | Alex |
|---|---|---|
| pooled failure states | 71 | 56 |
| candidates executed | 4544 | 3584 |
| real successes | 15 | 9 |
| **skills archived** | **2** | **2** |

Same headline, different skills. The original archived `episode_000040 @ offset 10` and
`episode_000008 @ offset 25`; Alex archived `episode_000023 @ offset 10` and
`episode_000008 @ offset 25` — the shared id is coincidental indexing, the `object_xy`
differs (`[0.176, 0.215]` vs `[0.182, 0.279]`). Hit rate per state: 2/71 = 2.8% against
2/56 = 3.6%.

The pool shrank because **the pooled failure count is the failure count**, and the Alex
policy fails less on the same 240 episodes:

| source directory | episodes | orig failures | Alex failures |
|---|---|---|---|
| f2s_final round 0 / 1 / 2 | 50 each | 13 / 17 / 16 | 7 / 12 / 18 |
| f2s_dev round 0 / 1 / 2 | 10 each | 3 / 2 / 2 | 3 / 2 / 1 |
| f2s_dev_cem round 0 / 1 / 2 | 10 each | 3 / 5 / 2 | 3 / 5 / 0 |
| fixed_policy seed 0 | 30 | 8 | 5 |
| **total** | **240** | **71** | **56** |

---

## 4. World model

Trained on the 50 round-0 evaluation episodes, identical 40/10 episode-level split.

| | original | Alex |
|---|---|---|
| best val MSE | 1.511e-4 | **1.110e-4** |
| beats constant baseline | yes | yes |

The Alex world model is **27% more accurate**, and better at every horizon — the
full-state multi-step MSE ratio widens from 0.75× at h=1 to 0.55× at h=20. It still
archived zero skills, which is an independent data point against world-model quality
being the bottleneck.

### 4.1 Object-position RMSE — one claim does not survive

| horizon | original | Alex | 50 mm success threshold |
|---|---|---|---|
| 1 | 3.9 mm | 3.4 mm | under |
| 5 | 18.3 mm | 15.9 mm | under |
| 8 | 27.9 mm | 24.4 mm | under |
| 12 | 39.4 mm | 34.9 mm | under |
| **16** | **50.1 mm (at threshold)** | **44.6 mm** | **under — claim does not reproduce** |
| 20 | 60.4 mm | 54.2 mm | exceeds (margin 10.4 → 4.2 mm) |

README_F2S.md calls h=16 "the decisive number": the model's own positional uncertainty
matching the threshold it is meant to resolve. On Alex h=16 sits comfortably below it.
The h=20 conclusion survives, with less room.

### 4.2 Contact dynamics — reproduces

Transport (contact-adjacent) is worse-predicted than approach (free space) at **every**
horizon. At h=20: 62.3 mm vs 51.7 mm on Alex, 73.6 mm vs 56.1 mm originally.

---

## 5. The capstone finding — the strongest reproduction

Recoverability predicted from features of the failure state alone, computed before any
candidate is generated, on a pool rebuilt from scratch (56 states, not 71):

| | original | Alex |
|---|---|---|
| **AUC** | 0.956 | **0.977** |
| Spearman ρ (task_progress ↔ success rate) | 0.245 (p = 3e-5) | 0.267 (p = 5.2e-5) |
| (state, offset) combinations | 284 | 224 |
| combinations with ≥1 real success | 7 | 6 |
| goal_error: success vs zero-success | 0.267 / 0.482 | 0.276 / 0.505 |
| task_progress: success vs zero-success | 0.733 / 0.518 | 0.724 / 0.495 |

The group means agree to the second decimal on a completely regenerated dataset. Whether
a failure is recoverable is set by how far the task had already progressed, not by which
correction is chosen — and §2.1 is what that costs in practice.

---

## 6. Reproducibility gaps found

Five, all in the same direction: the code assumes it runs once, on the machine that
produced the committed results.

### 6.1 A parameter the CLI never exposed

`discover_and_archive_skills` has always taken `use_cem`, but `run_evolution.py` never
exposed it, and the callee's default became `True` after `f2s_dev` and `f2s_final` were
produced with `False`. Re-running the recorded command today runs a different method
(2560–4800 candidates per round instead of 96–160). **Fixed** — `--use_cem` /
`--no_use_cem`, default unchanged.

### 6.2 A dataset-specific constant asserted as a correctness property

`analyze_recoverability_predictor.py` asserted `len(segments) == 71`. That number is a
function of the policy's success rate, so the capstone analysis could only ever run on
one machine's data. **Fixed** — replaced with the invariant the indexing depends on
(the pool must cover every `state_idx` the records reference); 71 still satisfies it.

### 6.3 Hardcoded output paths plus `ensure_fresh_dir`

`evaluate_candidate_ranking_per_state_offset_sweep.py`,
`analyze_recoverability_predictor.py` and the four Lift diagnostics hardcode their input
**and** output directories and wipe the output directory on start. Running them in place
deletes the committed originals, including the 4544-row `records.json` the capstone reads
and nine Lift result files. **Worked around, not fixed** — the reproduction runs them
unmodified inside a symlink sandbox outside the repository.

### 6.4 Two committed artifacts that no code produces

- `results/can/world_model_h20diag/multistep_eval/object_position_rmse_by_horizon.json` —
  the source of the h=16/h=20 table in §4.1. Added by commit `c58198d`; nothing in the
  repository writes it. The Alex column was obtained from
  `diagnose_world_model_error_by_stage.py`'s overall figures, which compute the same
  quantity.
- `results/can/candidate_ranking_per_state_offset_sweep/single_mode_cluster_model.pkl` —
  required by `--cluster_model_path` for every F2S evaluation. It is a degenerate
  `KMeans(n_clusters=1)` with identity normalisation, so `predict()` returns 0 for any
  input; the original file was reused, which carries no information from the original
  data.

---

## 7. Evaluation determinism

Prompted by `fixed_policy` seed 0 reading 73.3% in one job and 83.3% in another, from the
same checkpoint. Seven runs of the identical configuration — same checkpoint, seed 0, and
30 byte-identical initial states:

| run | success rate | mean episode length |
|---|---|---|
| evaluated in the training job's process | 73.3% (22/30) | 183.1 |
| independent run ×6 | **83.3% (25/30)** each | **162.9** each |

**Evaluation is deterministic across independent runs** — six runs agree to the episode
and to one decimal of mean length. The 10-point gap is a systematic offset for evaluation
running in the same process as training (GPU state after a training loop), not run-to-run
noise. The Alex main-table numbers are therefore real, not sampling variation.

Separately, the two runs that *do* differ diverge from timestep 0 at ~2.6e-5 in the
action output — floating-point nondeterminism compounding over a 400-step closed loop,
flipping 3 of 30 episodes.

---

## 8. Lift cross-task pressure test

| | original | Alex |
|---|---|---|
| Lift baseline success rate | 86.7% | 96.7% |
| failure states available | 4 | **1** |
| candidates executed / succeeded | 32 / 7 | 8 / 0 |
| skill archived | **no** (validation 0.2) | **no** (validation 0.1) |
| robust-CEM skill archived | **no** | **no** |
| ensemble variance ratio, 1 cm vs exact | 1.0011 | 0.9967 |

All three qualitative conclusions reproduce: the Lift skill fails Day-19 validation,
neighbourhood-robust CEM does not rescue it, and ensemble disagreement has no
discriminative power (ratio ≈ 1.0). The quantitative base is much thinner here — a 96.7%
policy leaves a single failure state in 30 episodes.

---

## 9. What this means for the paper

**Unaffected.** Every negative result: SOE 0%, Unguided Latent Repair 0%, the empty
archive under both candidate-generation paths, CEM finding nothing, Failure Replay
hurting, the Lift skill's brittleness, ensemble disagreement, and the capstone predictor
(stronger here than originally). These are the paper's substance and they are robust to a
change of machine, checkpoint and failure pool.

**Needs rewriting.** Any claim that F2S improves on the frozen baseline. The margin is two
episodes in one seed on one machine, and it is exactly zero here — with a mechanism, not a
mystery: the skills' 3 cm precondition never matches. This is better read as confirmation
of the capstone finding than as a failed reproduction.

**Needs restating.** The h=16 threshold claim in the world-model section.

**Worth adding.** A reproducibility note. Five separate obstacles stood between the
committed results and re-running them, and three still require a sandbox to work around.
For a paper whose contribution is a mechanistic negative result, the fact that the record
*can* be re-executed end to end — and what had to be repaired to get there — is part of
the claim.

---

## 10. Jobs

| job | stage | elapsed | state |
|---|---|---|---|
| 4197972 | world-model chain | 15:04 | completed (script exit 1 on a filename typo in the batch file after all four stages finished) |
| 4200886 | three evolution directories | 17:16 | completed |
| 4201035 | offset sweep, 3584 executions | 37:37 | completed |
| 4200859 | main table, 3 methods × 3 seeds | 2:21:11 | completed |
| 4201261 | f2s row + unseen evaluation | 1:08:33 | completed |
| 4201268 | failure replay: build, fine-tune, evaluate | 43:50 | completed |
| 4201266 | run-to-run variance, 5 repeats | 18:30 | completed |
| 4201269 → 4201653 | Lift baseline + 4 diagnostics | 9:42 → 3:56 | first attempt failed on a relative checkpoint path; rerun completed |

Logs are in `install_logs/alex_*.log`; batch scripts in `slurm/repro_*.sbatch`.
