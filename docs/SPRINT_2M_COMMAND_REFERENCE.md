# 2 m Sprint command reference

Use this as a command cookbook for the 2 m Sprint workflow. Commands assume the
`microduck-challenges` repo is the current directory unless a section says otherwise.
The challenge-specific rules and rationale live in
[`src/microduck_challenges/sprint_2m/AGENTS.md`](../src/microduck_challenges/sprint_2m/AGENTS.md).

## 0. Set up the terminal

```bash
cd /Volumes/sandisk_ssd/projects/microduck_challenges/microduck-challenges
export UV_CACHE_DIR=/private/tmp/microduck-uv-cache
export UV_LINK_MODE=copy
```

**Use for:** starting work in this checkout. The cache setting avoids the inaccessible
default uv cache seen on this Mac. `UV_LINK_MODE=copy` avoids hardlink warnings across
filesystems. Do not set `XDG_CACHE_HOME` for these commands; that can redirect
Hugging Face token lookup away from the authenticated HF home.

Once per checkout, sync the environment if needed:

```bash
uv sync --group workshop
```

**Use for:** installing the workshop evaluator and project dependencies. If the
environment already works, this is not needed before every command.

## 1. Check the active Hugging Face account

```bash
uv run hf auth whoami
```

**Use for:** verifying which account owns jobs and receives policy uploads before
launching or publishing.

## 2. Inspect the current experiment

```bash
git status --short
git show --stat --oneline 3dad9d31d
git show 3dad9d31d -- src/microduck_challenges/sprint_2m/challenge.toml

cat logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2/provenance.json
```

**Use for:** checking the working tree, seeing exactly what the `forward_reward` change
did, and verifying the run's seed, source commit, and clean/dirty provenance.

List the saved checkpoints:

```bash
run=logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2
find "$run" -maxdepth 1 -type f -name 'model_*.pt' -print \
  | sed -E 's#.*model_([0-9]+)\.pt#\1#' | sort -n
```

**Use for:** confirming which checkpoints exist before evaluating them. This run has
13: 0, every 250 through 2750, and 2999.

## 3. Run pre-training checks

```bash
uv run --group workshop pytest
```

**Use for:** checking the project before a training run, as required by the repo guide.

```bash
uv run train Mjlab-Sprint2m-MicroDuck --gpu-ids None --env.scene.num-envs 2 \
  --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke
```

**Use for:** a short CPU smoke test to catch launch/config errors before paying for a
full HF Job. It is not a meaningful policy evaluation.

## 4. Launch a Hugging Face training run

Example: repeat `forward_reward = 2.5` with training seed 3 to check whether the result
repeats. This starts a paid job; check the remaining experiment budget first.

```bash
uv run train Mjlab-Sprint2m-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.seed 3 \
  --agent.logger tensorboard \
  --agent.run-name forward25-seed3 \
  --hf-jobs \
  --namespace prithviraj-maurya \
  --ckpt-repo prithviraj-maurya/mjlab-sprint2m-microduck-forward25-seed3
```

**Use for:** launching training with the same settings and reward config as the seed 2
run, but a new random seed. The run name and checkpoint repo keep artifacts separate.
For a new reward hypothesis, first edit one knob, test, smoke-test, and commit that
hypothesis; use the new commit for the job. Never launch a paid job without checking
its budget and intended hardware/runtime cap.

## 5. Check and download a completed job

```bash
uv run hf jobs inspect <job-id> --namespace prithviraj-maurya

uv run hf download \
  prithviraj-maurya/mjlab-sprint2m-microduck-forward25-seed2 \
  --local-dir logs/rsl_rl
```

**Use for:** checking the job state/logs, then downloading its checkpoint tree. Replace
`<job-id>` and the repo name with the values printed for your job.

## 6. Read training curves

Open TensorBoard for the local Sprint runs:

```bash
uv run --no-sync tensorboard --logdir logs/rsl_rl/sprint_2m --port 6006
```

**Use for:** comparing training diagnostics. Open <http://localhost:6006>, select the
run, and start with `Train/mean_episode_length`,
`Episode_Termination/fell_over`, `Metrics/twist/error_vel_xy`,
`Metrics/twist/error_vel_yaw`, and `Episode_Reward/track_linear_velocity`. The reward
tag is weighted, so a higher curve alone is not proof of a faster policy. These two
HF jobs used `--no-wandb`; their charts are local TensorBoard data rather than W&B
histories. Stop the server with Ctrl-C.

Print the recent values for the main charts:

```bash
uv run --no-sync python -c '
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}).Reload()
for tag in sys.argv[2:]:
    print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-12:])
' logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2 \
  Train/mean_episode_length \
  Episode_Termination/fell_over \
  Episode_Reward/track_linear_velocity \
  Metrics/twist/error_vel_xy \
  Metrics/twist/error_vel_yaw
```

**Use for:** getting a compact numeric view without opening the TensorBoard UI. Remove
a tag from the command if that event file does not contain it.

## 7. Re-score a checkpoint locally

The full 13-checkpoint sweep for this run is recorded in
[`SPRINT_2M_EXPERIMENT_FORWARD25.md`](SPRINT_2M_EXPERIMENT_FORWARD25.md). To verify or
re-score a checkpoint, set `n` to its iteration number. This example re-scores the
current best, checkpoint 1000. The export and evaluation are local; they do not upload
or enter the policy or start a paid training job.

```bash
cd publish-evals
run=../logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2
n=1000

uv run --no-sync publish \
  --run "$run" \
  --checkpoint "$n" \
  --repo "local/microduck-eval-cp$n-verify" \
  --dry-run --allow-dirty

uv run --group workshop arena score \
  "publish-eval-cp$n-verify/policy.onnx" \
  --event sprint-2m 2>&1 | grep -E '^(scored at|seed [0-9]+:|score )'
```

**Use for:** locally exporting and evaluating one checkpoint. Change `n` and both
derived export paths together when checking another checkpoint. Run from `publish-evals/`
so generated export folders stay nested there. The filter keeps the
command speed, five seed results, and aggregate score while hiding evaluator startup
logs.

## 8. Replay a checkpoint in the Sprint workshop

```bash
uv run --group workshop workshop sprint-2m
```

**Use for:** opening the Sprint evaluator and replay UI. To watch the best-known
result, select run `2026-10-19_01-46-02_forward25-seed2`, checkpoint `1000`, then
select its evaluation replay for seed `2`. This replays that checkpoint under the
selected evaluation seed and is the closest local visual replay of the Sprint
evaluator.

## 9. Watch the exported ONNX in `microduck_rl`

First, export the current best checkpoint 1000 locally from the challenge repo:

```bash
cd publish-evals
uv run --no-sync publish \
  --run ../logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2 \
  --checkpoint 1000 \
  --repo local/microduck-eval-cp1000 \
  --dry-run --allow-dirty
```

**Use for:** producing `publish-evals/publish-eval-cp1000/policy.onnx` with the observation
normalizer baked in, without uploading it.

On macOS, `mjpython` needs a Python build with the shared library used by MuJoCo's
passive viewer. The viewer environment was set up separately using Homebrew Python
3.12, on the internal drive to avoid the repeated package-copy error on the external
drive:

```bash
cd /Volumes/sandisk_ssd/projects/microduck_challenges/microduck_rl

export UV_PROJECT_ENVIRONMENT=/Users/raj_mac_mini/Library/Caches/microduck-rl-mjpython-venv
export UV_CACHE_DIR=/private/tmp/microduck-uv-cache
export UV_LINK_MODE=copy

uv run mjpython -c "import mujoco; print(mujoco.__version__)"  # verify setup
```

**Use for:** selecting the already-installed viewer environment in each new terminal
and checking that MuJoCo imports. To create or refresh it, run
`uv sync --python /opt/homebrew/opt/python@3.12/bin/python3.12` first. This separate
environment leaves the existing `microduck_rl/.venv` in place.

Then launch the best-known policy in the interactive MuJoCo viewer at its best-known
command speed:

```bash
uv run mjpython scripts/infer_policy.py \
  --walking /Volumes/sandisk_ssd/projects/microduck_challenges/microduck-challenges/publish-evals/publish-eval-cp1000/policy.onnx \
  --new-cmd-obs --lin-vel-x 0.70
```

**Use for:** watching checkpoint 1000 walk at 0.70 m/s. This opens the `microduck_rl`
MuJoCo scene and uses its BAM actuator setup. It is a visual rehearsal, not the Sprint
Arena lane; `infer_policy.py` has no Arena evaluation seed option, so this does not
replay seed 2 or reproduce official timing. Use section 8 for a specific checkpoint
and Arena seed. Keyboard controls are printed in the terminal; `Q` quits. To view
another checkpoint, export it in section 9 and update the `--walking` path to that
export.

## 10. Publish and enter a selected checkpoint

First confirm the experiment commit is pushed to your fork:

```bash
git status --short
git status -sb
```

**Use for:** checking for uncommitted files and whether `main` is ahead of
`origin/main`. Preserve unrelated notes; don’t stage them with the training change by
accident.

If needed, push the training commit:

```bash
git push origin main
```

**Use for:** making the source commit available for the policy's code link. The
`forward25` commit `3dad9d31d` was already pushed when this reference was written.

When you have selected a checkpoint and explicitly intend to submit it, publish a
public repo and enter it. This example uses a **new repo name** for the current best,
checkpoint 1000,
so it preserves any policy already published as `microduck-forward25-seed2`:

```bash
uv run --no-sync publish \
  --run logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2 \
  --checkpoint 1000 \
  --repo prithviraj-maurya/microduck-forward25-seed2-cp1000 \
  --no-private --enter
```

**Use for:** uploading the selected ONNX publicly and entering it in the Sprint
leaderboard. Change the checkpoint number if you select another checkpoint.
`--no-private --enter` makes the policy public and submits it. Confirm the
account first with `uv run hf auth whoami`. If you intentionally want to replace the
policy in an existing repo, add `--force`; otherwise choose a fresh repo name as above.

## Quick distinction

| Command type | Costs HF Job budget? | Uploads or submits publicly? |
|---|---:|---:|
| `train ... --hf-jobs` | Yes | No, unless separately published |
| `publish ... --dry-run` + `arena score` | No training charge | No |
| `workshop sprint-2m` / `mjpython ...` | No training charge | No |
| `publish ... --no-private --enter` | No training charge | Yes |
