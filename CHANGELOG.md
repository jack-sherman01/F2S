# Changelog

All notable changes to F2S are recorded here.

## Instructions for contributors

**When to add an entry**
Add an entry for any commit that meaningfully changes behaviour, adds a feature, removes something, or fixes a bug. Skip trivial commits (typo fixes, comment edits, formatting).

**Where to add it**
Always append at the **top**, immediately below `## [Unreleased]`. Do not edit existing entries.

**Format**
```
## YYYY-MM-DD — <short title (≤ 72 chars)>

**Commits:** `24c956a`, `e73334c`

### <Module or area changed>
- What changed and why (not just "updated X" — explain the consequence)
- Breaking changes must be marked **[BREAKING]**
- New public APIs must be marked **[API]**
```

**Scope**
Group bullets by module/file area. Skip files that only had mechanical renames or import updates — focus on behaviour and interface changes.

**Experimental results are behaviour**
This repository's product is an experimental record, so a run that changes what a number
means is as reportable as a code change. When an entry cites a result, cite the file it
lives in, and say which machine and checkpoint produced it.

---

## [Unreleased]

## 2026-09-08 — Re-run the whole experimental record on Alex; F2S's margin is zero

**Commits:** `968067a`, `8666bc0`, `5086344`

Every experiment in `README_F2S.md` re-executed on NHR@FAU Alex from a checkpoint
retrained there, ~6 h GPU across 9 Slurm jobs. Full comparison in
`SOE/REPRODUCTION_ALEX.md`; all output under `*_alex` directories, so nothing recorded on
the original `/data/heng/F2S` machine is overwritten and every number stays comparable.

### Results (`SOE/results/**_alex/`, `SOE/REPRODUCTION_ALEX.md`)
- **[BREAKING]** **F2S no longer improves on the frozen baseline.** Alex gives
  83.3 / 80.0 / 70.0 across seeds 0-2 for *both* F2S and Fixed Policy — equal seed by
  seed, and equal on the unseen split (43% each). The cause is mechanical, not
  statistical: `skills_used` is empty in all 90 F2S evaluation episodes, so the archived
  skills' 3 cm spatial precondition never matched a failure and F2S reduces to Fixed
  Policy. Re-reading the original numbers with this in hand, the published +2.2 point
  mean margin (74.4% vs 72.2%) is **two episodes in seed 0**; seeds 1 and 2 were already
  identical there. Any claim that F2S beats the baseline has to be withdrawn or restated
  as what the capstone finding predicts: skills validate only at near-misses, so their
  applicability region is the 3 cm neighbourhood Day-19 validation actually tested, and a
  different checkpoint fails elsewhere.
- Every negative result reproduces, several more strongly. SOE 0% and Unguided Latent
  Repair 0% land seed-for-seed. Failure Replay's mean is 65.6% on both machines, and
  against Alex's stronger baseline the "naive replay hurts" conclusion widens from 6.6 to
  12.2 points. Three evolution rounds archive nothing on either candidate-generation path
  (10,560 candidates, 66 real executions on the CEM path). The offset sweep archives
  exactly 2 Day-19-validated skills again — different skills, from a pool rebuilt from
  scratch.
- The capstone recoverability predictor is the strongest reproduction in the set:
  **AUC 0.977 against 0.956**, Spearman 0.267 (p = 5.2e-5) against 0.245, with the
  success-group and zero-success-group means for `goal_error` and `task_progress` agreeing
  to the second decimal — on 224 (state, offset) combinations rebuilt independently of the
  original 284.
- **The world model's h=16 claim does not reproduce.** README_F2S.md calls it "the
  decisive number": 50.1 mm object-position RMSE, exactly at the 50 mm success threshold.
  Alex measures **44.6 mm**, comfortably under. h=20 still exceeds (54.2 mm vs 60.4 mm),
  with the margin down from 10.4 to 4.2 mm. The Alex world model is 27% more accurate
  overall (val MSE 1.11e-4 vs 1.51e-4) and better at every horizon — and archived zero
  skills anyway, which is an independent data point against world-model quality being the
  bottleneck.
- The failure pool shrank 71 → 56 states because the pooled failure count *is* the failure
  count, and the Alex policy fails less on the same 240 episodes. Per-directory breakdown
  in the report.
- Lift reproduces qualitatively — skill rejected by Day-19 validation, robust CEM does not
  rescue it, ensemble variance ratio ≈ 1.0 — on a much thinner base: a 96.7% Lift policy
  leaves 1 failure state in 30 episodes, against 4 originally.

### Evaluation determinism (`SOE/results/Can/fixed_policy_variance/`)
- The same checkpoint, seed and 30 byte-identical initial states, evaluated five more
  times, returns **83.3% every time** with mean episode length agreeing to one decimal.
  Evaluation is deterministic across independent runs; there is no run-to-run sampling
  noise to explain results away with. The one 73.3% reading came from the evaluation that
  ran *in the same process as the training job* that produced the checkpoint — a
  systematic ~10-point offset, worth knowing before quoting a number produced that way.
  Where two runs do differ, they diverge from timestep 0 at ~2.6e-5 in the action output
  and compound over the 400-step loop, flipping 3 of 30 episodes.

### Tooling (`SOE/scripts/run_evolution.py`, `SOE/scripts/analyze_recoverability_predictor.py`)
- **[API]** `--use_cem` / `--no_use_cem` on `run_evolution.py`.
  `discover_and_archive_skills` has always taken `use_cem`, but the CLI never exposed it,
  and the callee's default became `True` *after* `results/can/f2s_dev/` and
  `results/can/f2s_final/` were produced with `False`. Re-running the recorded command
  today therefore runs a different method — 2560-4800 candidates per round instead of
  96-160. With the flag restored the per-round counts reproduce exactly (`f2s_dev` 48/32,
  `f2s_dev_cem` 960/1600). Default unchanged, so no existing invocation moves.
- `analyze_recoverability_predictor.py` asserted `len(segments) == 71`, the pooled
  failure-state count of one particular run. That number is a function of the policy's
  success rate, so the capstone analysis could only ever run on the original machine's
  data. Replaced with the invariant the code actually depends on — the pool must cover
  every `state_idx` the records reference — which the original 71-state pool still
  satisfies.

### Reproduction infrastructure (`SOE/slurm/repro_*.sbatch`, `SOE/configs/`)
- Eight batch scripts, one per stage of the dependency chain, each calling the committed
  analysis scripts with their originally recorded parameters.
- Three of them run their analysis script inside a symlink sandbox outside the repository.
  `evaluate_candidate_ranking_per_state_offset_sweep.py`,
  `analyze_recoverability_predictor.py` and the four Lift diagnostics hardcode both their
  input directories and their output directory, and call `ensure_fresh_dir` on the latter
  — so running them in place **deletes the committed originals**, including the 4544-row
  `records.json` the capstone analysis reads and nine Lift result files. Worked around,
  not fixed; the scripts themselves run unmodified.
- `configs/soe_can_lowdim_failure_replay_alex.json` differs from the committed config only
  in three absolute paths; the committed one still points `resume_ckpt` at the old
  machine's `2026-08-29` checkpoint directory, which does not exist on this cluster.
- Two committed artifacts have no generating code anywhere in the repository:
  `results/can/world_model_h20diag/multistep_eval/object_position_rmse_by_horizon.json`
  (the source of the h=16/h=20 table) and
  `results/can/candidate_ranking_per_state_offset_sweep/single_mode_cluster_model.pkl`
  (required by `--cluster_model_path` for every F2S evaluation; a degenerate
  `KMeans(n_clusters=1)` with identity normalisation, so reusing the original file carries
  no information from the original data).

## 2026-09-07 — Alex baseline retrained; branches merged

**Commits:** `2205edf`, `7fd598f`, `c3973e4`

### Baseline (`SOE/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/`)
- The Day-3 Can baseline retrained on an A40: 500 epochs in 22 min, final train loss
  0.009564 against the original machine's 0.0095. Every downstream artifact — world
  models, failure pools, offset sweeps, the three-seed table — depends on this checkpoint,
  and the original did not survive the machine move (`*.ckpt` is gitignored).

### Repository policy (`CLAUDE.md`)
- Commit messages and committed content carry no AI/assistant attribution — no
  `Co-Authored-By` trailers, no "generated with" lines, no links to AI-hosted output.
  Applies to every contributor and tool.

## 2026-09-04 — Migrate to the NHR@FAU Alex cluster

**Commits:** `acaa82e`

### Environment (`SOE/env.sh`, `SOE/slurm/`)
- **[BREAKING]** `env.sh` rewritten for Alex. Both `$HOME` (100 GB quota) and `$WORK` were
  over quota for this user, so the conda env, package and pip caches, torch and
  HuggingFace caches, `TMPDIR`, datasets, checkpoints and results all live inside the
  `f2s` hpc-workspace on `/anvme`. The old file is kept at
  `install_logs/env.sh.old_machine_data_heng`. Sourcing the previous `env.sh` on this
  cluster activates nothing.
- Login nodes have no GPU: anything needing CUDA goes through Slurm. `slurm/` holds the
  job scripts. The `rtxpro6k` partition is unusable — Blackwell `sm_120` predates the
  pinned `torch 1.13.0+cu117` wheel.
- Everything gitignored was lost in the move: the Day-3 checkpoint, every world-model
  `best_model.pt`, every raw `episodes/` directory (including the 71-state pooled failure
  set), the RoboMimic datasets and the `dependencies/` clones. Datasets and dependencies
  are re-fetchable from the pinned commits; the rest has to be regenerated.

### Acceptance (`SOE/install_logs/alex_smoketest_4148553.log`)
- Day-1.3 acceptance passed on an A40: 2-epoch training loss 1.070094 → 1.000066, the same
  two numbers as the original machine's Day-1 run, so the seeded data pipeline and model
  init reproduce exactly.
