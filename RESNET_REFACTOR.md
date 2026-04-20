# ResNet-CIFAR10 Portfolio Refactor Plan

Status tracker for elevating this repo from "clean tutorial" to "fundamentals
executed carefully." Target audience on review: engineer at Nvidia Robotics /
Wayve / Figure AI / Tesla. This repo's job is **not** to win you an interview —
other, robotics-adjacent repos do that — its job is to **not be a liability**
and to signal that you handle fundamentals with care.

## Honest framing

- **Current headline (84.6%) is the #1 liability.** He et al. ResNet-20 is
  91.25%. Any repo claiming a "ResNet from scratch" result below paper baseline
  reads as incomplete to a senior engineer. Fix this before anything else.
- **Topic is unsurprising.** CIFAR-10 classification is the most common DL
  exercise. Execution quality is the only lever for signal here.
- **Do not over-engineer.** Docker, Hydra, hyperparameter sweeps, distributed
  training — none of these add signal for a single-GPU CIFAR-10 repo. They add
  noise and read as cargo-culting. Staying within scope is itself a signal.
- **Scope lock.** Everything in this plan should land in ≤ ~15 focused commits
  over ~2 weekends of work. If an item grows beyond that, cut it.

## Model-choice triage legend

Each task is tagged with two labels:

- **Category:** `mechanical` | `structural` | `content` | `decision`
- **Suggested model:** `sonnet` | `opus`

Rules of thumb used:

| Category   | Meaning                                                        | Default model                                 |
| ---------- | -------------------------------------------------------------- | --------------------------------------------- |
| mechanical | Rote edits, boilerplate, formatting, renames, obvious splits   | sonnet                                        |
| structural | Reorganizing layout, wiring modules together, standard pattern | sonnet                                        |
| content    | Writing docs, tests, READMEs, explanations                     | sonnet (opus for README strategy / narrative) |
| decision   | Tradeoffs, hyperparameter strategy, eval design, what to cut   | opus                                          |

Use sonnet by default. Reach for opus only when the task requires a judgment
call whose cost (wasted GPU hours, bad framing, hidden-bias eval) is larger
than the cost of the extra tokens.

---

## Workstream overview

Phases are ordered by leverage per hour. Do not reorder — every later phase
assumes the prior one landed. Each phase maps to one or more commits.

| #   | Phase                             | Cat.       | Model  | Est. time | Commit(s) |
| --- | --------------------------------- | ---------- | ------ | --------- | --------- |
| 1   | Training recipe → paper baseline  | decision   | opus   | 4–6 h GPU | 1         |
| 2   | Seeding & determinism             | mechanical | sonnet | 45 min    | 1         |
| 3   | Config system (argparse + YAML)   | structural | sonnet | 1 h       | 1         |
| 4   | Logging (JSONL + TensorBoard)     | structural | sonnet | 1.5 h     | 1         |
| 5   | Checkpoint format + resume        | mechanical | sonnet | 45 min    | 1         |
| 6   | Mixed precision + throughput log  | mechanical | sonnet | 30 min    | 1         |
| 7   | Evaluation script                 | content    | opus*  | 1.5 h     | 1         |
| 8   | Tests                             | content    | sonnet | 1.5 h     | 1         |
| 9   | Project layout + pyproject + ruff | structural | sonnet | 1 h       | 1         |
| 10  | CI (ruff + pytest CPU)            | mechanical | sonnet | 30 min    | 1         |
| 11  | README rewrite                    | content    | opus   | 2 h       | 1         |
| 12  | Artifacts: curves, conf. matrix   | content    | sonnet | 45 min    | 1         |
| 13  | Final pass: dead code, polish     | mechanical | sonnet | 30 min    | 1         |

`*` opus on the design of what metrics to report (per-class accuracy,
confusion matrix, calibration?), sonnet on the implementation.

Total: ~12 h human + ~4–6 h GPU.

---

## Phase 1 — Training recipe → paper baseline

**Category:** decision. **Model:** opus.

**Goal:** headline accuracy ≥ 91.5% (He et al. ResNet-20 is 91.25%). Stretch:
≥ 93% by going to ResNet-32 depth (same param budget class).

**Why opus:** hyperparameter strategy is a judgment call. Wrong calls cost
GPU-hours. You want one pass that lands.

**Recipe to implement (opinionated, do not deviate without a reason):**

- Optimizer: SGD, momentum=0.9, nesterov=True, weight_decay=5e-4
- LR: 0.1 initial, **linear warmup** for 5 epochs, then cosine annealing to 0
- Epochs: 200 (paper uses 64k iters ≈ 164 epochs; round up)
- Batch size: 128
- Loss: CrossEntropy with label_smoothing=0.1
- Augmentation: keep RandomCrop(32, padding=4) + HFlip — do not add Cutout/Mixup
  yet; that muddies the "did my recipe work" signal
- Normalization: keep CIFAR-10 mean/std (already correct)
- Mixed precision: torch.amp autocast + GradScaler
- Deterministic seeding (see Phase 2)

**Architecture decision:** keep 6-block design for v1 so the accuracy delta is
clearly attributable to the *recipe*, not a bigger model. After hitting
baseline, optionally add a `--depth` flag and report ResNet-20/32/56 as rows
in the results table.

**Deliverable:** one training run, curves saved, final acc in README.

**Acceptance:** test acc ≥ 91.5%. If you land below, do not ship — diagnose
(LR schedule, warmup present, label smoothing wired, autocast actually on).

---

## Phase 2 — Seeding & determinism

**Category:** mechanical. **Model:** sonnet.

Add `src/resnet_cifar10/utils/seeding.py` with a `set_seed(seed: int)` that
sets:

- `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `os.environ["PYTHONHASHSEED"]`
- `torch.use_deterministic_algorithms(True, warn_only=True)` behind a flag

Wire a `--seed` CLI flag (default 42). Also seed the DataLoader worker init
via `worker_init_fn` and a generator. README must state: "results reproduce
bitwise on the same GPU architecture with seed=42."

---

## Phase 3 — Config system

**Category:** structural. **Model:** sonnet.

Keep the `Config` dataclass. Add argparse that can override any field and an
optional `--config path/to/run.yaml` that loads a YAML into the same dataclass.
Save the resolved config next to every checkpoint (`run/<timestamp>/config.yaml`).

**Do not** adopt Hydra or OmegaConf. For a repo this size it is strictly
over-engineering and reads as resume-padding.

Remove the `num_workers=0 # required on Windows` comment — default to 4, add a
CLI flag.

---

## Phase 4 — Logging

**Category:** structural. **Model:** sonnet.

Three sinks, all wired from a single `Logger` object:

1. **stdout** — human-readable per-epoch line
2. **JSONL** — `run/<ts>/metrics.jsonl`, one line per epoch with all scalars.
   This is what feeds the curve-generation script in Phase 12.
3. **TensorBoard** — `run/<ts>/tb/`. Log: train_loss, test_loss, test_acc,
   lr, images/sec, grad_norm (optional).

Do not pull in W&B. Requires an account; hiring engineers may not follow a
link into a third-party dashboard. Static artifacts in the repo are better.

---

## Phase 5 — Checkpoint format

**Category:** mechanical. **Model:** sonnet.

Current code saves `model.state_dict()` only. Replace with a dict:

```python
{
    "epoch": int,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_acc": float,
    "config": asdict(cfg),
    "torch_rng_state": torch.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state_all(),
}
```

Add `--resume path/to/ckpt.pth`. Save both `best.pth` and `last.pth`.

---

## Phase 6 — Mixed precision & throughput

**Category:** mechanical. **Model:** sonnet.

- `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` around forward+loss
- `GradScaler` around backward/step
- Log images/sec per epoch (use `time.perf_counter`, exclude first batch warmup)

Report throughput in the README results table. Hiring engineers do read this.

---

## Phase 7 — Evaluation script

**Category:** content. **Model:** opus for metric design, sonnet for code.

Create `scripts/evaluate.py --checkpoint best.pth`. Outputs:

- Top-1 accuracy (headline)
- Per-class accuracy (bar chart → `artifacts/per_class.png`)
- Confusion matrix (heatmap → `artifacts/confusion_matrix.png`)
- JSON dump of all metrics → `artifacts/eval.json`

**Opus decision:** should we include calibration (ECE), adversarial robustness,
or param count / FLOPs? Recommendation: include param count + FLOPs
(via `fvcore` or a hand-rolled counter — fvcore adds a heavy dep, consider
`thop` or skip and report params only). Skip ECE and adversarial — scope creep.

---

## Phase 8 — Tests

**Category:** content. **Model:** sonnet.

Minimum viable:

- `tests/test_model.py`:
  - forward pass shape: (B,3,32,32) → (B,10)
  - parameter count within tolerance of expected
  - residual block dimension-match path triggers when stride≠1 or channels differ
- `tests/test_dataset.py`:
  - Train/test loader yield correct tensor shapes and dtypes
  - Normalization actually applied (sample mean within tolerance of 0)
- `tests/test_training.py`:
  - Smoke test: 1 epoch on a 256-sample subset, loss decreases
  - Determinism: same seed → same loss to N decimals

Keep fast (<30s total on CPU). These run in CI.

---

## Phase 9 — Project layout

**Category:** structural. **Model:** sonnet.

Move to src layout:

```
resnet-cifar10/
├── pyproject.toml
├── README.md
├── RESNET_REFACTOR.md
├── Makefile
├── src/
│   └── resnet_cifar10/
│       ├── __init__.py
│       ├── model.py
│       ├── dataset.py
│       ├── trainer.py
│       ├── config.py
│       ├── logging.py
│       └── utils/
│           └── seeding.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
├── configs/
│   └── resnet20.yaml
└── artifacts/
```

`pyproject.toml`: build-system = hatchling, project metadata, deps, ruff
config, pytest config. Drop `requirements.txt` or keep as a minimal
pin-generated file (use `uv pip compile`). Add `.python-version` (3.11).

Makefile targets: `train`, `eval`, `test`, `lint`, `fmt`, `clean`.

---

## Phase 10 — CI

**Category:** mechanical. **Model:** sonnet.

`.github/workflows/ci.yml`:

- Trigger: push to main, PR
- Jobs: lint (ruff check + ruff format --check), test (pytest -q)
- Python 3.11, CPU-only torch install
- Green badge in README

Do not add a GPU CI job. Out of scope and expensive.

---

## Phase 11 — README rewrite

**Category:** content. **Model:** opus.

**This is the single most-read artifact in the repo.** Spend proportional time.

Structure (hiring engineer should grasp the project in 90 seconds):

1. **Headline (one line):** "ResNet-20 reimplemented from scratch in PyTorch,
   reproducing He et al. (2015) CIFAR-10 result (91.X% top-1) on a single
   consumer GPU."
2. **Badges:** CI, Python version, license.
3. **Results table:**
   | Model | Params | FLOPs | Top-1 | He et al. | Δ |
   (with your number alongside the paper's)
4. **Training curve image** (generated, committed to `artifacts/`).
5. **Confusion matrix image.**
6. **Reproduce this result** — 3 lines:
   ```
   pip install -e .
   python scripts/train.py --config configs/resnet20.yaml --seed 42
   python scripts/evaluate.py --checkpoint runs/.../best.pth
   ```
7. **Hardware used:** GPU model, VRAM, wall-clock time per epoch, total.
8. **Architecture** — keep short, 3-4 bullets. Link to paper.
9. **What this project demonstrates** — 2-3 honest bullets. Example: "careful
   reproduction of a paper baseline on fixed hardware with deterministic
   seeding" — not "deep understanding of residual connections."
10. **Limitations** — 2-3 bullets. Examples: "no depth scan beyond ResNet-20,"
    "no adversarial or OOD robustness eval," "single-seed result; no
    mean±std across runs" (unless you do N=3).
11. **References:** He et al. 2015.

**Remove from current README:** the "Key concepts" section (reads as a
textbook restating obvious facts) and the "Further improvements" list (signals
the project was left unfinished).

---

## Phase 12 — Artifacts

**Category:** content. **Model:** sonnet.

Script `scripts/plot_curves.py`: reads `metrics.jsonl`, produces
`artifacts/training_curves.png` (train loss + test acc on twin axes).

Commit the images. They are the fastest way a reviewer confirms the number
in the README is real.

---

## Phase 13 — Final pass

**Category:** mechanical. **Model:** sonnet.

- Remove any dead code, unused imports, TODO comments
- `ruff check .` clean, `ruff format .` clean
- `pytest` green
- Verify README reproduce-command actually works from a clean clone
- Tag a `v1.0` release. A tagged release reads as "this is finished."

---

## Commit strategy

One commit per phase. Commit messages in imperative present, tight subject
line, body explains *why* not *what*:

```
feat: train ResNet-20 to 91.X% with SGD+cosine recipe

Replaces the Adam + StepLR recipe (84.6% final) with SGD momentum 0.9,
weight decay 5e-4, linear warmup over 5 epochs, cosine annealing to 0
over 200 epochs, label smoothing 0.1, and torch.amp mixed precision.
Matches He et al. 2015 ResNet-20 baseline.
```

Prefer `feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `ci:` /
`chore:` prefixes. Consistency here is itself a small signal.

---

## Progress log (update as you go)

| Phase | Status        | Commit SHA | Notes |
| ----- | ------------- | ---------- | ----- |
| 1     | ✅ done        |            |       |
| 2     | ✅ done        |            |       |
| 3     | 🟨 in progress |            |       |
| 4     | ⬜ todo        |            |       |
| 5     | ⬜ todo        |            |       |
| 6     | ⬜ todo        |            |       |
| 7     | ⬜ todo        |            |       |
| 8     | ⬜ todo        |            |       |
| 9     | ⬜ todo        |            |       |
| 10    | ⬜ todo        |            |       |
| 11    | ⬜ todo        |            |       |
| 12    | ⬜ todo        |            |       |
| 13    | ⬜ todo        |            |       |

Legend: ⬜ todo · 🟨 in progress · ✅ done · ❌ abandoned (note reason)

---

## Out of scope (explicit NO list)

These were considered and rejected. Noted here so future-you does not
rediscover and waste time on them.

- **Docker** — adds no signal for a `pip install` project; adds a liability
  if the image goes stale.
- **Hydra / OmegaConf** — over-engineering for ~15 hyperparameters.
- **W&B** — requires account, hiring engineers may not follow third-party
  links; static artifacts committed to the repo serve the same purpose.
- **DDP / multi-GPU** — CIFAR-10 fits on one consumer GPU; going distributed
  reads as cargo-culting.
- **Hyperparameter sweep** — one well-chosen recipe beats 50 random runs.
- **ONNX / TensorRT export** — not the story this repo tells. Save for a
  deployment-themed repo.
- **Adversarial / OOD eval** — scope creep. Belongs in a separate robustness
  repo if it matters to you.
- **Cutout / Mixup / RandAugment** — useful for pushing past 93%, but adds
  a confounder on the "did my base recipe work" question. Only add after
  baseline is locked, and only if you report ablation.

---

## When to stop

You are done when:

- Headline accuracy ≥ 91.5%, reported with hardware and exact command.
- A clean clone → 3 commands reproduces the result on the same GPU.
- CI is green.
- Both images (training curves, confusion matrix) are committed and visible
  in the README.
- The README tells a reviewer in 90 seconds: what, how, result vs baseline,
  limitations.

Past that point, move on. Additional polish has lower portfolio ROI than
**starting the next, robotics-adjacent repo.**
