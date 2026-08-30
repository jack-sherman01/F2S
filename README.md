# F2S: From Failures to Skills

**World-Model-Guided Open-Ended Self-Improvement for Robot Manipulation**

Robot policy self-improvement typically relies on successful trajectories,
fixed tasks, and manually designed exploration mechanisms. Although
failure trajectories contain rich diagnostic information, they are usually
treated only as negative samples and never systematically turned into
reusable skills.

**F2S** is a failure-driven, open-ended self-evolution framework for
simulated robot manipulation, built on top of the public
[SOE](https://github.com/EricJin2002/SOE) codebase. For each failure
trajectory, F2S:

1. identifies the failure stage and type,
2. generates corrective candidates in SOE's latent action space,
3. predicts their outcomes with a lightweight action-conditioned world
   model,
4. filters out unsafe or high-uncertainty candidates,
5. and stores validated behaviors as reusable skills with preconditions,
   effects, and associated failure modes.

The skill archive is then retrieved from and updated across future
rollouts, so the policy accumulates targeted, failure-specific recovery
behavior over time rather than relying only on broad, undirected
exploration.

All experiments are simulation-only, run on a single GPU, and use
low-dimensional simulator state rather than pixel-level world models, to
keep the study controlled and reproducible.

## Repository layout

```
F2S/
├── SOE/                  # vendored SOE codebase + the F2S implementation
│   ├── f2s/               # f2s Python package (logging, failure analysis,
│   │                       #   world model, candidates, safety, skills,
│   │                       #   evolution loop)
│   ├── scripts/            # CLI entry points for every pipeline stage
│   ├── configs/            # experiment configs
│   ├── results/            # committed metrics/configs/figures (raw
│   │                       #   episode data and checkpoints are gitignored
│   │                       #   and regenerable from a config + seed)
│   └── README_F2S.md       # detailed execution log: exact commands run,
│                            #   environment/versions, every experiment's
│                            #   real results, and known open items
└── LICENSE
```

`SOE/README_F2S.md` is the primary technical log for this project — it
records the exact setup commands, dependency versions, every experiment
actually run with real (not fabricated) results, bugs found and fixed
along the way, and a clear account of what is implemented versus still
open.

## Status

Environment setup, a reproduced SOE baseline, and the full F2S pipeline
(failure extraction/clustering, world model, candidate generation and
ranking, safety filtering, skill archive/retrieval, and a multi-round
evolution loop) are implemented and have been run end to end against the
real simulator. See `SOE/README_F2S.md` for current results, including
what has and has not yet worked, and for the next steps toward the full
experimental matrix (multiple seeds, baselines, ablations, and
generalization to unseen configurations).

## License

MIT — see [LICENSE](LICENSE).
