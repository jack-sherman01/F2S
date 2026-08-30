# F2S Execution Log (README_F2S.md)

This file records the exact commands run to set up and reproduce SOE, and the
concrete environment/version information, as required by the F2S proposal
(`private/proposal.tex`, Days 1-3). It is updated incrementally as the
project proceeds; nothing here is invented — every value is filled in only
after the corresponding command has actually been run.

## Machine / hardware notes (deviation from proposal defaults)

- Proposal assumes a single NVIDIA RTX 3090. This machine instead has
  **2x NVIDIA RTX A5000 (24GB each)**. We use a single GPU
  (`cuda:0`, device index 0) throughout, matching the proposal's
  "single GPU process at a time" rule. A5000 and 3090 are both
  Ampere, 24GB, so this is treated as a compatible substitution.
- The root filesystem (`/`) on this machine has only ~6GB free. All
  conda environments, pip/HF/torch caches, the SOE clone, datasets,
  checkpoints, and results therefore live under `/data/heng/F2S`
  (172GB free at project start), which is symlinked to
  `/home/heng/work/F2S` for convenience. The git repository itself
  (`.git`) also lives on `/data`.
- Every shell command for this project should start with:
  `source /data/heng/F2S/SOE/env.sh`
  which activates the `f2s` conda env and points pip/conda/HF/torch
  caches at `/data/heng/.cache` and `/data/heng/miniconda3/pkgs`
  instead of the default (root-fs) locations.

## Day 1: Environment setup

### Day 1.1: Conda environment

The proposal's generic default (`python=3.10`) was overridden in favor of
the SOE README's own explicit recommendation ("We recommend using Python
3.8 for better compatibility with the dependencies"), since the proposal's
Day-1 rule is to follow the official README exactly rather than guessing.

```bash
conda create -n f2s python=3.8 -y
```

### Day 1.2: SOE installation (commands copied verbatim from SOE README.md)

```bash
conda activate f2s
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

mkdir dependencies && cd dependencies
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ../..

cd dependencies
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout 9273f9cce85809b4f49cb02c6b4d4eeb2fe95abb
pip install -e .
cd ../..
```

Note: `requirements.txt` pins `torch==1.13.0` / `torchvision==0.14.0`,
older than a plain `conda install pytorch ... pytorch-cuda=12.1` build.
In practice `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch
-c nvidia` on this machine resolved to a **CPU-only** build twice in a row
(this machine's conda is configured with `conda-forge` as the default
channel, and the solver kept substituting a `cpu_generic` pytorch build
even with `-c pytorch -c nvidia` given on the command line; forcing
`--override-channels -c pytorch -c nvidia` instead made the solver
unsatisfiable, because base libs like `jpeg`/`libpng` are only available
via conda-forge/defaults on this box). Since `requirements.txt` pins the
exact version we need anyway (`torch==1.13.0`), the actually-executed
sequence was:

```bash
python -m pip install torch==1.13.0 torchvision==0.14.0 \
    --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt   # torch/torchvision already
                                             # satisfied at the pinned
                                             # version, left untouched
```

This installs the exact pinned torch/torchvision versions with real CUDA
11.7 GPU wheels (verified: `torch.cuda.is_available() == True`, a real
matmul was run on `cuda:0`), which is a more reliable path to the authors'
actual pinned dependency than the generic conda command, given this
machine's channel configuration.

pytorch3d's source build additionally required a matching **CUDA 11.7
nvcc + dev headers** (the system-wide CUDA toolkit here is 12.2, which
`torch.utils.cpp_extension` refuses to build against for a torch 11.7
binary). Installed narrowly into the `f2s` env only, not system-wide:

```bash
conda install -n f2s -c "nvidia/label/cuda-11.7.0" \
    cuda-nvcc cuda-cudart-dev cuda-cccl \
    libcusparse-dev libcublas-dev libcurand-dev libcusolver-dev \
    libcufft-dev libnvjitlink -y
```

With `CUDA_HOME=/data/heng/miniconda3/envs/f2s`, `TORCH_CUDA_ARCH_LIST=8.6`
(Ampere, matching the RTX A5000) and `FORCE_CUDA=1`, `pip install -e .`
in `dependencies/pytorch3d` then built cleanly.

<!-- FILLED_VERSIONS_START -->
Recorded versions (filled in automatically as install steps complete):

- Python version: 3.8.20
- PyTorch version: 1.13.0+cu117
- torch.cuda.is_available(): True
- CUDA version (torch.version.cuda): 11.7 (torch wheel); nvcc in env: 11.7 (system-wide nvcc is 12.2, driver 535.230.02 / CUDA 12.2, not used for the build)
- GPU: NVIDIA RTX A5000 (device 0)
- robosuite version: 1.4.1
- robomimic version / commit: 0.3.1 @ 9273f9cce85809b4f49cb02c6b4d4eeb2fe95abb (pinned, per README)
- mujoco version: 3.2.3
- pytorch3d version: 0.7.9
- SOE git commit (vendored): 4d4f069e322f12f6f62bcef68d87d73f6d86fcb1
- Full pinned dependency snapshot: see `environment_f2s.yml` and `requirements_f2s.txt`
<!-- FILLED_VERSIONS_END -->

### Day 1.3: Official minimal example — blocked, then worked around

**Blocker found:** SOE's own `.gitignore` (in the vendored `SOE/.gitignore`)
excludes `simulation/config_template/`, `src/config/`, and
`realworld/config/` from their public GitHub repo. The README's documented
pipeline command,

```bash
python run_full_multi_round.py --dataset datasets/can/ph/image_v141.hdf5 \
    --output_dir out/can_soe_multi_round/ --used_demo core_20 \
    --config can_soe --seeds 233 2333 23333 233333 --cuda_device 0 1 2 3 \
    --noise_scale 2.0
```

reads `simulation/config_template/can_soe.json`, which **does not exist**
anywhere in the released code (`git ls-remote` shows only a single
`master` branch, no tags, no GitHub releases, no issues on the repo as of
this writing). This is a gap in the authors' public release, not a
mistake on our side — the code that *consumes* a config (`train_single_gpu.py`,
`src/policy/dp.py`, `src/dataset/robomimic_v2.py`, `run.py`) is fully
present, only the pre-made JSON config instances are missing.

**Resolution:** rather than guess at the multi-round *orchestration*
command, a config JSON was hand-authored by reading the config schema out
of the consuming code directly (`configs/soe_can_lowdim_dev.json`) —
this is the smallest faithful entry point into the real, unmodified SOE
training code (`train_single_gpu.py`), using the **low-dim** RoboMimic
`Can` observations already downloaded (`simulation/datasets/can/ph/low_dim_v141.hdf5`,
200 demos; using the `20_percent_train` filter key, 36 demos, as the
smallest official demo-count split shipped in the dataset's own `mask/`
group — chosen as a fast Day-1 smoke test, matching the spirit of
`core_20` in the README's own example without inventing a new mask).
`MultiImageObsEncoder` (used inside `DP`) natively supports
`type: "low_dim"` observation keys, so no image rendering / pixel
world-model work is introduced here, consistent with the proposal's scope.
The dataset's low-dim obs keys/shapes (`object` (14,), `robot0_eef_*`,
`robot0_gripper_*`, `robot0_joint_*`) and action shape (7,) were read
directly out of the hdf5 file rather than assumed.

See `results/can/soe/seed_0/round_0/official_demo.log` for the actual run
output.

**Result:** `python train_single_gpu.py --config configs/soe_can_lowdim_dev.json`
(cwd `SOE/src`) ran 2 epochs on 36 demos (`20_percent_train`, 4141
training windows), loss 1.070 -> 1.000, and saved
`policy_epoch_1_seed_0.ckpt`, `policy_epoch_2_seed_0.ckpt`,
`policy_last.ckpt`, a loss curve plot, and `config.json` under
`results/can/soe/seed_0/round_0/logs/soe_can_lowdim_dev/<timestamp>/`.
`python run.py --agent <ckpt> --config <same config> --n_rollouts 2
--try_times 1 --seed 0 --dataset_path .../rollouts.hdf5 --dataset_obs`
(cwd `SOE/simulation`) then launched the real RoboSuite/MuJoCo
`PickPlaceCan` environment, ran the trained policy (diffusion-policy
action-chunk inference, `action_dim=7`, `horizon=20`), executed both the
plain-rollout and SOE exploration-rollout code paths, ran every episode to
completion (`horizon=400`), and wrote a 4-episode rollout HDF5 (`actions`
shape `(400,7)`, `states` shape `(400,71)`). Success rate was 0/4, which
is expected and fine for a 2-epoch smoke test — this run's purpose is
solely to prove every pipeline stage (data -> policy -> optimizer ->
checkpoint -> simulator -> action execution -> episode termination ->
output file) is wired correctly on real, unmodified SOE code before Day 3's
full baseline training run.

**Day 1 acceptance test: PASSED**
1. Official demo (adapted per the missing-config finding above) completed
   without exception. ✅
2. PyTorch accesses the GPU (`NVIDIA RTX A5000`, substituting for the
   proposal's RTX 3090). ✅
3. Episodes terminate (4/4 rollouts ran to `horizon=400` and produced
   `dones`/`rewards`/`states`). ✅
4. Repository commit and full environment are recorded (`soe_commit.txt`,
   `environment_f2s.yml`, `requirements_f2s.txt`, this file). ✅

## Status: Days 4-21 (logging, failure analysis, world model, F2S pipeline)

Built continuously after Day 1 rather than paced literally one calendar
day at a time (the day numbers are the proposal's dependency ordering,
not a schedule). Everything below is real, runnable code exercised
against the actual SOE/robosuite/robomimic simulator -- nothing is a
mocked stub -- and has been committed incrementally with its own
acceptance-test log where the proposal specifies one.

- **f2s/common**: shared JSON/npz I/O, seeding, and the four canonical
  data schemas (episode, failure_segment, candidate, skill).
- **f2s/logging**: `EpisodeLogger` (method-independent episode storage)
  and `metrics.py`. `scripts/selftest_logging.py` -- PASSED.
- **f2s/failure**: structured (non-visual) failure feature extraction for
  RoboMimic Can, failure-time/type/stage detection via documented
  heuristics over the logged state trajectory (there is no raw contact
  sensor in the low-dim dataset, so "collision"/"object_drop" are proxy
  labels -- see the module docstrings for exactly what each one means),
  and K-means failure-mode clustering.
- **f2s/world_model**: the proposal's exact residual-MLP architecture,
  trained on real transitions from logged episodes. In every run so far
  (including the two evolution-loop smoke tests below) it has beaten the
  constant-state baseline by roughly 30x on held-out validation data.
- **f2s/candidates**: latent-space candidate generation (Gaussian /
  single-dimension / historical-skill perturbations of SOE's own DP
  readout vector, decoded with SOE's own unmodified diffusion action
  decoder), world-model-based scoring/ranking, and real-simulator
  execution + Day-19 cross-configuration validation.
- **f2s/safety**: deterministic hard-constraint filter (collision proxy,
  Panda joint limits, joint-velocity limit, predicted object drop,
  invalid actions). `scripts/selftest_safety_filter.py` -- PASSED (5/5
  hand-built trajectories classified correctly).
- **f2s/skills**: `Skill` dataclass, `SkillArchive` (Day 20.1 archive
  rule + Day 20.2 duplicate removal), `retrieve()`.
  `scripts/selftest_skill_archive.py` -- PASSED.
- **f2s/evolution/loop.py**: the full round loop -- evaluate (with online
  skill retrieval on a detected task-progress stall) -> extract/cluster
  failures -> generate/rank/filter/execute/validate/archive candidates ->
  retrain the world model on all transitions seen so far.
  `scripts/run_evolution.py` drives N rounds of this against a real
  trained checkpoint.

### Known, deliberate scope limitation: no policy-weight fine-tuning

The proposal's Figure 1 loop includes a "Policy Update" step. This
codebase's evolution loop retrains the **world model** every round but
does **not** fine-tune the DP policy's weights -- the skill archive is
used purely at *evaluation time* (the stall-detector + skill-retrieval
rollout in `rollout_with_skills`), not folded back into the base policy.
This was a scoping decision made under the 4-week/single-session time
budget, not an oversight: it keeps every reported number traceable to
either (a) the frozen baseline checkpoint or (b) the skill archive, with
no risk of silently conflating the two. `--method success_only` and
`--method failure_replay` in `scripts/run_method.py` are consequently
refused with `NotImplementedError` rather than reporting a fabricated
number for a method that isn't actually implemented -- both require this
same retraining loop.

### Evolution-loop smoke tests (real code, tiny scale, not the final experiment)

Run against `policy_epoch_300` (strong) and `policy_epoch_100` (weak) of
the Day-3 baseline training below, `configs/f2s_smoketest.yaml`
(4 episodes/round, 1 round):
- Strong checkpoint: 3/4 success, 1 failure -- correctly took the
  "too few failures to cluster" early-exit path (needs >= 2).
- Weak checkpoint: 4/4 failure -- ran the complete chain: 4 failure
  segments extracted, K=2 clusters fit, 32 real candidates generated
  (8 Gaussian + 8 single-dim per state x 2 states x 2 modes), 0 rejected
  by the safety filter, 4 executed in the real simulator, all correctly
  rejected at the archive's success-rate threshold (a 20-step correction
  from a mid-episode failure state, generated by a policy trained for
  only 100 epochs on 36 demos, is not expected to solve the task --
  this is the filter working as intended, not a bug).
- World model in both runs beat the constant-state baseline by ~30x.
- 0 crashes in either run.

## Days 3, 21, 23: real experiments on the trained baseline, and a key finding

With the full 500-epoch baseline trained (Day 3: loss 1.07 -> 0.0095;
`results/can/soe/seed_0/round_0/`):

- **fixed_policy** (no exploration, no skills), 30 episodes: **73.3%
  success**. A genuine reproduced SOE baseline.
- **soe** (SOE's own `--enable_exploration --noise_scale 2.0`, the
  README's recommended range), 30 episodes: **0% success**. Real, not a
  bug: CADS noise is injected into the diffusion conditioning on *every*
  action query for the *entire* episode by default (`tau1=0, tau2=1`),
  which is a data-collection mechanism (generate diverse candidates for
  curation), not meant for direct autonomous deployment. See
  `results/Can/soe/seed_0/round_0/`.
- **F2S**, 3 rounds x {10, 50} episodes/round (`results/can/f2s_dev/`,
  `results/can/f2s_final/`): world model beat the constant-state baseline
  by 30-90x in every single round; the safety filter rejected a real,
  varying fraction of candidates each round (not a rubber stamp); but
  **0 skills were ever archived**, at either scale.

Chasing why led to two real bugs (fixed, both regression-tested against
the live simulator -- see `scripts/selftest_validator_perturbations.py`):
`env.obj_body_id`/`env.sim` needed to be `env.env.*` (robomimic's env
wrapper doesn't proxy attribute access), and the "object-position
validation" config was resampling a *brand-new random* placement rather
than a small offset near the original failure state. Neither bug,
however, was the actual reason 0 skills got archived -- both live
downstream of a candidate ever succeeding once, which never happened.

**The actual finding** (`scripts/diagnose_candidate_success_rate.py`):
of up to 320 candidates directly executed across 5-10 real failure
states, **0 ever succeeded** -- even after fixing a real upstream bug
where every timeout-type failure's `failure_time` defaulted to the
episode's literal last frame (`find_stall_time()` now identifies the
actual moment task progress last improved, which is far more meaningful
and was confirmed to produce much more varied, sensible failure times).
Undirected Gaussian / single-dimension perturbation of a 59-dim diffusion-
policy readout vector, decoded by a decoder never trained for recovery,
appears to have a very low hit rate for landing in a narrow "this
actually solves the task" basin at the proposal's candidate budget
(M=16-64 tested). This is reported as a genuine negative result, per the
proposal's own Day-27 rule ("if the world model does not improve ranking,
report the negative result explicitly" -- the same principle applies
here to candidate generation). The natural next lever, not yet
implemented: replace pure random sampling with a guided search (e.g. CEM
using the world model's own predicted-success score as the objective)
before executing candidates in the real simulator.

## What's real vs. what's still open, for anyone picking this up

**Done and verified against the real simulator, not stubbed:** full SOE
env/dependency setup; a real trained baseline policy (73.3% success);
method-independent logging/metrics; failure extraction (with a corrected
stall-based failure-time) and K-means clustering; a lightweight world
model that beats the constant-state baseline by 30-90x every round
tested; latent-space candidate generation via SOE's own diffusion
decoder; world-model-based candidate ranking; a safety filter with a
5/5-correct unit test that also does real rejection work in real runs;
a skill archive/retrieval implementation with its own passing acceptance
test; an evolution loop that runs 3 real rounds end to end with no
crashes at both dev and final episode scale.

**Genuinely open (not fabricated, not silently skipped):**
- No skill has yet been successfully archived (see finding above) --
  H3 (skill archive improves generalization) cannot be evaluated until
  at least one skill exists to test.
- Policy-weight fine-tuning ("Policy Update" in the proposal's Figure 1)
  is not implemented; the skill archive is used only at evaluation time.
  `success_only`/`failure_replay` baselines are consequently refused
  (`NotImplementedError`) rather than faked.
- Only seed 0 has been run for any method; the proposal's final
  experiments require seeds {0,1,2}.
- Ablations (Day 24), unseen-configuration evaluation (Day 25), and the
  reproduction-from-clean-directory test (Day 28) have not been run yet.
- `soe` (0% success, by design of the mechanism at noise_scale=2.0) has
  not been tried at the README's lower recommended noise scales, which
  would likely change that number substantially.
