# Sprint 2 m experiment: forward reward 2.0 → 2.5

This note records the second 2 m Sprint run and gives the commands to reproduce the
workflow, inspect the learning curves, and judge the exported policy. Run commands from
the repository root.

For a reusable command-by-command cookbook with each command's use case, see
[`SPRINT_2M_COMMAND_REFERENCE.md`](SPRINT_2M_COMMAND_REFERENCE.md).

On this Mac, if `uv` cannot open its default cache, set an accessible cache for this
terminal before running the commands below. Do not redirect `XDG_CACHE_HOME`, because
Hugging Face authentication uses the default Hugging Face home.

```bash
export UV_CACHE_DIR=/private/tmp/microduck-uv-cache
```

## What changed

The hypothesis was: **raising the forward velocity tracking reward from 2.0 to 2.5 may
help the duck follow faster forward commands while still finishing 5/5 trials.**

Only `params.forward_reward.value` in
[`challenge.toml`](../src/microduck_challenges/sprint_2m/challenge.toml) changed. The
training environment maps this knob to the `track_linear_velocity` reward weight in
[`env.py`](../src/microduck_challenges/sprint_2m/env.py). The robot, reward formula,
other reward weights, PPO settings, environment count, and iteration count were not
changed.

| Setting | Control | Experiment |
|---|---:|---:|
| `forward_reward` / `track_linear_velocity` weight | 2.0 | 2.5 |
| Environments | 4,096 | 4,096 |
| Training iterations | 3,000 | 3,000 |
| Training seed | 2 | 2 |
| Run name | `seed2-control` | `forward25-seed2` |

The experiment commit is `3dad9d31d` (`Sprint 2m: raise forward tracking reward from
2.0 to 2.5`). Its `provenance.json` says `dirty: false`, so the run records a clean
source tree.

## Full checkpoint sweep results

| Checkpoint | Score | Finishes | Command | Replay seed |
|---:|---:|---:|---:|---:|
| 0 | 50.000 s | 0/5 | 1.10 m/s | — |
| 250 | Not captured | 3/5 (two falls) | Not captured | — |
| 500 | 4.680 s | 5/5 | 0.55 m/s | 5 |
| 750 | 4.976 s | 5/5 | 0.80 m/s | 5 |
| 1000 | **3.884 s** | **5/5** | **0.70 m/s** | **2** |
| 1250 | 4.024 s | 5/5 | 0.70 m/s | 2 |
| 1500 | 4.064 s | 5/5 | 0.65 m/s | 3 |
| 1750 | 4.424 s | 5/5 | 0.65 m/s | 5 |
| 2000 | 4.376 s | 5/5 | 0.60 m/s | 2 |
| 2250 | 4.544 s | 5/5 | 0.65 m/s | 5 |
| 2500 | 4.252 s | 5/5 | 0.65 m/s | 4 |
| 2750 | 4.328 s* | 5/5 | 0.65 m/s | — |
| 2999 | 4.220 s | 5/5 | 0.65 m/s | 4 |

\*Checkpoint 2750's 4.328 s is the mean of its five reported seed times; the aggregate
score line was not captured.

For comparison, the seed 2 control scored 5.048 s, 5/5, at 0.55 m/s at its best
reported checkpoint. Checkpoint 1000 is 1.164 s (about 23%) faster in the local
evaluator while finishing 5/5, and it found a faster winning command. Checkpoint 1250
is also strong at 4.024 s. Later checkpoints are generally slower in this sweep,
showing why the final training iteration is not automatically the best policy. This
supports the hypothesis for this training run, but does not establish repeatability
across training seeds; the Arena is the final measure.

### Repeat run: training seed 3

The same `forward_reward = 2.5` configuration was trained with seed 3. Its full
checkpoint sweep is:

| Checkpoint | Score | Finishes | Command | Replay seed |
|---:|---:|---:|---:|---:|
| 0 | 50.000 s | 0/5 | 0.20 m/s | 1 |
| 250 | 27.096 s | 3/5 | 0.25 m/s | 5 |
| 500 | 6.912 s | 5/5 | 0.30 m/s | 1 |
| 750 | 4.072 s | 5/5 | 0.65 m/s | 1 |
| 1000 | **3.864 s** | **5/5** | **0.65 m/s** | **4** |
| 1250 | 3.880 s | 5/5 | 0.65 m/s | 3 |
| 1500 | 4.152 s | 5/5 | 0.65 m/s | 1 |
| 1750 | 4.468 s | 5/5 | 0.65 m/s | 5 |
| 2000 | 4.552 s | 5/5 | 0.60 m/s | 1 |
| 2250 | 4.268 s | 5/5 | 0.60 m/s | 2 |
| 2500 | 4.296 s | 5/5 | 0.60 m/s | 2 |
| 2750 | 4.300 s | 5/5 | 0.65 m/s | 4 |
| 2999 | 4.332 s | 5/5 | 0.65 m/s | 2 |

At checkpoint 1000, the per-seed times printed to two decimals were 3.88, 3.86,
3.86, 3.86, and 3.86 s; the aggregate mean was 3.864 s. The fastest displayed time
is shared by seeds 2–5, while the evaluator selected seed 4 for its replay. This
checkpoint is 0.020 s faster locally than seed 2's checkpoint 1000 (3.884 s), with
5/5 finishes in both runs. The winning commands differed (0.65 vs 0.70 m/s), so treat
this very small timing difference as a close repeat, not a meaningful speed
breakthrough. Both training runs peaked at checkpoint 1000; later checkpoints were
slower.

Seed 3 provenance records commit `3dad9d31d` but `dirty: true`: the local recipe seed
in `challenge.toml` had been changed while the training command also explicitly passed
`--agent.seed 3`. The score sweep is valid as local evaluation evidence, but resolve
the dirty provenance before considering this run's policy for publication.

## Commands used for this experiment

### Check the code and the saved run

```bash
git show --stat --oneline 3dad9d31d
git show 3dad9d31d -- src/microduck_challenges/sprint_2m/challenge.toml

cat logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2/provenance.json
find logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2 \
  -maxdepth 1 -name 'model_*.pt' -print | sort -V
```

The first command shows the commit summary; the second shows the precise parameter
change. Provenance records the command, seed, source commit, clean/dirty state, and
package revision used for training.

### Download a completed Hugging Face Job

```bash
uv run hf auth whoami
uv run hf jobs inspect <job-id> --namespace prithviraj-maurya

uv run hf download \
  prithviraj-maurya/mjlab-sprint2m-microduck-forward25-seed2 \
  --local-dir logs/rsl_rl
```

The job ID identifies the job page. The checkpoint repository contains the run files.
The folder for this run is:
`logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2`.

### Open TensorBoard plots

Start TensorBoard from the repository root:

```bash
uv run --no-sync tensorboard \
  --logdir logs/rsl_rl/sprint_2m \
  --port 6006
```

Open <http://localhost:6006> in a browser. In the **Scalars** tab, inspect:

- `Train/mean_episode_length`: moves toward 1,000 for full 20-second episodes.
- `Episode_Termination/fell_over`: should trend down toward zero.
- `Episode_Reward/track_linear_velocity`: weighted training reward for following the
  commanded velocity.
- `Metrics/twist/error_vel_xy` and `Metrics/twist/error_vel_yaw`, if present: lower
  tracking errors are generally better.

Use the run selector to compare `2026-10-19_01-46-02_forward25-seed2` with
`2026-10-18_06-10-52_seed2-control`. Keep the x-axis as training step/iteration and
compare the overall trend, not one noisy point. Look for falls or episode length
getting worse at curriculum transitions (around iterations 500, 750, 1000, 1250,
1500, and 2000). Stop TensorBoard with Ctrl-C in its terminal when finished.

Important: `Episode_Reward/track_linear_velocity` is the **weighted** reward. Raising
its weight from 2.0 to 2.5 changes the scale of this curve by 25% even if the duck's
actual speed tracking did not improve. Treat it as a training diagnostic, not proof of
a faster policy. The Arena evaluator's time, finishes, and command speed are the
performance measures.

### What the saved curves show

Approximate values near selected iterations, read from the two TensorBoard event files:

| Signal | Control at 1250 | `forward25` at 1250 | What it suggests |
|---|---:|---:|---|
| Mean episode length | 971 / 1,000 | 963 / 1,000 | Both usually survive most of the episode; this is similar. |
| `fell_over` | 0.375 | 0.167 | Fewer falls in this sample at that point, but noisy and not uniformly better later. |
| Weighted velocity reward | 0.736 | 1.025 | Higher, partly because its configured weight is 25% larger. |
| XY velocity error | 1.219 | 1.092 | Slightly lower in this logged sample; consistent with improved tracking, not proof by itself. |
| Yaw velocity error | 1.121 | 1.232 | Slightly worse here; the change did not improve every training metric. |

At iteration 2999, both runs' mean episode lengths were still near 1,000 (975 for the
control, 982 for `forward25`) and both had zero `nan_state` terminations. The forward25
run's weighted velocity reward was higher (1.203 vs 0.875), while its XY error was
similar (1.027 vs 1.040) and yaw error somewhat higher (0.979 vs 0.876). These curves
do not show a simple across-the-board improvement. The clearest evidence is the Arena
evaluator result at checkpoint 1000. A repeat training seed will tell us how robust
that improvement is.

To print recent scalar values without opening the plots:

```bash
uv run python -c '
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

If a tag is absent, remove it from the command; available tags can be viewed in
TensorBoard.

## Reading the W&B charts

The two Hugging Face Jobs for `seed2-control` and `forward25-seed2` were launched with
`--no-wandb`, so their plots are not in W&B. Their TensorBoard event files contain the
same training scalars. For a future run with W&B enabled, the chart names below are the
ones to start with. In W&B, compare runs using the same x-axis and keep smoothing low
while diagnosing spikes.

| Chart/tag | Meaning | How to read it for this sprint |
|---|---|---|
| `Train/mean_episode_length` | Average episode length across the parallel training environments | Toward 1,000 is good: this task's training episode is at most 1,000 steps. It is not the Arena's 2 m race time. |
| `Episode_Termination/fell_over` | Episodes ending because the duck fell | Lower is generally better. Expect noisy fluctuations; compare trends across hundreds of iterations. |
| `Episode_Termination/time_out` | Episodes reaching the simulation's time limit | Often means it stayed up for the episode; this is not the Arena's 50-second race timeout. Read alongside episode length and falls. |
| `Episode_Termination/out_of_terrain_bounds` | Duck left the training terrain | Lower is better. |
| `Episode_Termination/nan_state` | Invalid numerical simulation state | Should remain zero; nonzero values warrant investigation. |
| `Metrics/twist/error_vel_xy` | Difference between commanded and actual planar velocity | Lower generally means closer speed tracking. It averages training conditions, so it is not the evaluator's winning command or time. |
| `Metrics/twist/error_vel_yaw` | Difference in commanded and actual yaw rate | Lower generally means better yaw-rate tracking. The sprint commands zero yaw. |
| `Episode_Reward/<term>` | Weighted contribution of one reward term per completed episode | Positive terms encourage behavior; penalty terms should be nonpositive. A rise is useful only if task performance also improves. |
| `Train/mean_reward` | Overall training reward | A broad learning signal, not a race score. It can rise while an important behavior stays weak. |
| `Loss/value`, `Loss/surrogate` | PPO critic and policy optimization losses | Training diagnostics. There is no simple rule that “lower is better”; look for instability or divergence, not a target score. |
| `Loss/entropy`, `Policy/mean_std` | Policy randomness / action spread | These often fall as the policy settles. A rapid collapse early can mean it stopped exploring; they are not direct quality measures. |
| `Loss/learning_rate` | Current optimizer learning rate | Can change because this task uses an adaptive schedule. A change alone does not mean the robot improved or got worse. |
| `Curriculum/*` | Training difficulty or reward-weight schedules | Steps or jumps may be intentional curriculum transitions. Compare behavior before and after the transition. |
| `Perf/total_fps`, `Perf/collection_time`, `Perf/learning_time` | Training throughput and time spent collecting/updating | Useful for diagnosing job speed or hardware, not walking quality. |

For a first pass, pin just these five charts: `Train/mean_episode_length`,
`Episode_Termination/fell_over`, `Metrics/twist/error_vel_xy`,
`Metrics/twist/error_vel_yaw`, and `Episode_Reward/track_linear_velocity`. Then inspect
the other reward terms only when you have a specific question, such as whether the duck
is staying upright or lifting its feet.

Read the x-axis as PPO training iterations, not seconds. One iteration collects data
from 4,096 environments and then updates the policy. W&B's smoothing can hide spikes;
use the unsmoothed or lightly smoothed view when checking curriculum transitions. Do
not compare curves from different seeds as if they were identical trajectories; use
them to compare broad patterns and rely on the Arena evaluator for the result.

### Score every checkpoint locally

This exports each checkpoint using the publish exporter, then scores the resulting ONNX
with the Sprint evaluator. `--dry-run` does not upload or enter a policy.

```bash
cd publish-evals
run=../logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2

for n in $(ls "$run" | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
  uv run publish --run "$run" --checkpoint "$n" \
    --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 &&
  echo "checkpoint $n: $(uv run --group workshop arena score \
    publish-eval/policy.onnx --event sprint-2m | \
    grep -E '^(scored at|score )' | paste -sd' ' -)"
done
```

For each checkpoint, record the score, finishes, and command speed. Choose a 5/5
checkpoint first, then the lowest time. The best checkpoint can occur before the last
iteration.

### Publish and enter the selected checkpoint

Only do this after choosing the checkpoint and confirming the source commit is pushed
to your fork. This makes the policy public and submits it to the Arena:

```bash
git status --short
git push origin main

uv run publish \
  --run logs/rsl_rl/sprint_2m/2026-10-19_01-46-02_forward25-seed2 \
  --checkpoint 1000 \
  --repo prithviraj-maurya/microduck-forward25-seed2-cp1000 \
  --no-private --enter
```

If `git status --short` reports unrelated work, preserve and resolve that work before
publishing; do not add it to the experiment commit by accident. Change `1000` only if
another checkpoint is selected. `--no-private` makes the Hub repo public; `--enter`
submits it to the leaderboard.

## How to learn from the plots and scores

Reinforcement learning has two different kinds of evidence here:

1. **Training curves explain what learning looked like.** Episode length, falls, and
   reward terms help spot instability, collapse, or a reward that is not improving.
2. **Checkpoint evaluation measures the task.** The Arena evaluator checks whether the
   exported policy can finish the sprint, how long it takes, and which forward command
   it handles best.

Checkpoint 1000 is the best measured policy in both training runs: seed 2 scored
3.884 s at 0.70 m/s and seed 3 scored 3.864 s at 0.65 m/s, each with 5/5 finishes.
This repeat supports the result across training seeds, although the 0.020 s difference
is too small to claim a real timing gain. Checkpoint 1250 was also close in both runs;
later checkpoints were slower. Compare curve trends to ask *why* performance peaked
around iteration 1000.

## End-of-experiment review: what to check

Treat these as three layers of evidence. The Arena result decides whether the policy
performed well; training curves help explain how it learned; the live leaderboard tells
you whether that performance improves your current rank.

### 1. Check training health and learning signals

Review the whole run in TensorBoard, especially around curriculum changes and the
checkpoint that later scored best. Compare trends over hundreds of iterations instead
of judging one noisy point.

| Statistic | What to look for | What it tells you |
|---|---|---|
| `Train/mean_episode_length` | Trends toward 1,000 steps | The robot survives longer in the training environment; it is not the 2 m race time. |
| `Episode_Termination/fell_over` | Trends down and stays low | Fall frequency during training. Compare with the evaluator's 5-seed finishes. |
| `Episode_Termination/out_of_terrain_bounds` | Low or zero | Whether the robot frequently leaves the training terrain. |
| `Episode_Termination/nan_state` | Remains zero | Non-finite simulation states indicate a serious training problem. |
| `Metrics/twist/error_vel_xy` | Trends lower | Planar velocity tracking across training commands; not the evaluator's score. |
| `Metrics/twist/error_vel_yaw` | Trends lower | Yaw-rate tracking. The Sprint command asks for zero yaw. |
| `Episode_Reward/<term>` | Read each term, including penalties | A weighted reward can rise because its weight changed; it does not prove a faster policy. |
| `Train/mean_reward`, PPO losses, entropy | Look for collapse or instability | Diagnostics only. There is no single target value that means the robot is race-ready. |

Ask: did the curves improve before the best checkpoint? Did falls or tracking error
worsen after a curriculum change? Did total reward rise while the task-relevant metrics
stayed flat? For this run, checkpoint 1000 scored best, even though later training
continued to iteration 2999.

### 2. Compare exported checkpoints with the evaluator

For every candidate checkpoint, preserve these values:

| Value | Why to record it |
|---|---|
| Checkpoint iteration | Identifies the exact saved policy; the final checkpoint is not automatically best. |
| Best command speed (`scored at`) | Shows which speed the evaluator found best for that policy. A higher speed with a worse time is not an improvement. |
| Aggregate score / mean time | Primary local performance comparison for this sprint. Lower is better. |
| Finished seeds out of 5 | Reliability gate. Prefer 5/5 before optimizing time; a fall is heavily penalized. |
| Each seed's time or failure | Shows consistency and reveals a policy that succeeds only on some starts. |
| Spread between fastest and slowest seed | A quick robustness check; wide spread means the average hides sensitivity. |
| Replay seed | Lets you inspect a representative run visually. It is not the training seed. |

The sprint evaluator sweeps command speeds, then evaluates promising speeds across
five fixed evaluation seeds and scores the mean time. For this project, compare policies
first by 5/5 completion, then by the lowest mean time. Also record the command speed;
do not use speed alone as the success measure. Local scores are useful for selecting
checkpoints, but the Hugging Face Arena result is authoritative because simulation
hardware can change timings.

### 3. Check the leaderboard result

Before submission, inspect the current [2 m Sprint Arena leaderboard](https://huggingface.co/spaces/pollen-robotics/microduck-arena)
and record the score and finishes of the entries you want to beat. The Sprint board
ranks by score, then number of seeds finished, then submission time. After an entry is
evaluated, record the Arena's official score, command speed, finishes, rank, and the
policy repo URL. Check the board again later because other entries can change your
position.

The optional API command from the Sprint guide prints the current event rows:

```bash
curl -s https://pollen-robotics-microduck-arena.hf.space/api/events/sprint-2m/leaderboard
```

Keep the local evaluator result and official Arena result in separate columns. Never
claim a leaderboard improvement from a local score alone.

## Design the next experiments

Use a small, controlled hill-climb loop. Each paid training run should answer one
question; a policy that improves on a measured baseline earns the next experiment.

1. **Write the question and hypothesis before editing.** State the one change, the
   reason it might help, and the expected observable result. Example: “Does increasing
   `forward_reward` from 2.0 to 2.5 help the Sprint policy finish faster at 0.65–0.70
   m/s without reducing 5/5 reliability?”
2. **Choose one variable.** Change one reward weight, algorithm setting, or other
   supported factor. Keep the environment, iterations, hardware, and other settings
   fixed. Do not change several knobs together; if the score changes, you would not
   know which change caused it.
3. **Set the success rule in advance.** For this challenge, require 5/5 finishes and
   a lower evaluator mean time than the baseline. Treat a faster command speed as
   useful only when its evaluated time and completion rate also improve. If the score
   is similar or inconsistent, call the result inconclusive rather than a win.
4. **Control run-to-run randomness.** Training seeds 2 and 3 both produced their best
   measured policy at checkpoint 1000 (3.884 s and 3.864 s, respectively), supporting
   the repeatability of this result. Training seeds affect learning; evaluator seeds
   1–5 test a saved policy.
5. **Use the same evaluation procedure.** Score the same checkpoint schedule, use the
   same event, and record all five evaluation results. Pick the best eligible
   checkpoint, not the last. Replay it to understand failures or unusual gait.
6. **Keep a baseline and a result ledger.** Record commit, hypothesis, changed value,
   training seed, HF job ID and cost, run name, best checkpoint, speed, score, finishes,
   per-seed results, and official Arena rank. Preserve the TensorBoard logs and
   `provenance.json` so the run can be reproduced.
7. **Only keep changes that survive the check.** If the new run is slower, less
   reliable, or fails to repeat, keep the better checkpoint/configuration as the
   baseline and write down what the test taught you. Avoid stacking a second change
   onto an unverified result.

The `forward_reward = 2.5` result has now repeated across training seeds 2 and 3, with
the best checkpoint at iteration 1000 in both runs. The next step is to compare the
clean seed 2 policy with the live Arena result before spending more training budget.
For another training experiment, first inspect the current leaderboard and choose a
limitation to target; then write a one-variable hypothesis with a success rule.
Before starting a paid HF Job, check the remaining budget, run the required tests and
smoke test, and follow the challenge guide's approval rule for paid jobs.

### Fill-in report for each experiment

Copy this into the hypothesis notes after each run:

```text
Hypothesis / one changed variable:
Baseline and success rule:
Commit / config / training seed:
HF job ID / actual cost:
Best checkpoint / command speed:
Local score / finished seeds:
Per-seed times or failures / replay seed:
Training curves: episode length, falls, XY and yaw error, reward-term notes:
Official Arena score / finishes / rank / policy URL (if submitted):
Conclusion: supported / refuted / inconclusive, and why:
Next single change or repeat:
```



Design each hypothesis to test **one explanation** for the policy’s current limitation. Write down what you’ll change, why it might help, what result would support the idea, and what result would count against it.

A useful template:

> **Hypothesis:** Changing *[one knob]* from *[current value]* to *[new value]* will improve *[specific behavior]*, because *[reason]*. I’ll compare the best checkpoint’s Arena score with the baseline, requiring 5/5 finishes. If the score is unchanged or worse, I’ll reject this setting.

## Start with a reliable baseline

Your two recent `forward_reward = 2.5` runs both had their best checkpoint around iteration 1000, with local scores of 3.884 s and 3.864 s. Those are close, so use them as the baseline rather than treating the 0.020 s difference as a real improvement.

For each experiment, record:

- The code commit, training seed, and run name.
- The best checkpoint, its score, finish count, and commanded speed.
- The five seed times, including falls.
- Whether the result beats the baseline on the same local evaluator.

Compare **best checkpoint to best checkpoint**. Keep all five seeds finished before treating a faster time as a win. Local scores are useful for choosing candidates; the Arena leaderboard is the final measure.

## A sensible order for testing knobs

### 1. Test the speed-tracking reward

This is the most direct knob for the sprint objective.

> **Hypothesis:** Raising `forward_reward` from 2.5 to 3.0 will make the policy track higher speed commands more strongly, improving its best sprint time while preserving 5/5 finishes.

Keep the run length, seed, and other settings fixed. A faster score with falls is not a win; first check whether all five seeds finish. If it improves, repeat the same setting with another training seed before relying on it.

### 2. Test foot-lift reward if the gait looks like it drags or shuffles

> **Hypothesis:** Raising `step_reward` from 3.0 to 3.5 will produce clearer steps and reduce foot dragging, improving completion reliability or speed.

Only try this if replay or logs give you a reason to suspect foot clearance is limiting the policy. If the robot already lifts its feet well, changing this knob may add effort without helping the sprint.

### 3. Test upright reward if failures show instability

> **Hypothesis:** Raising `upright_reward` from 2.0 to 2.5 will improve balance and increase the number of completed seeds.

This might also make the gait more cautious. Judge it by finish reliability and time together, not by upright behavior alone.

## Then consider training settings

Keep these separate from reward experiments:

- **Training seed:** Repeat the same setup with another seed to estimate run-to-run variation. This tests reliability, not a new reward idea.
- **Actor size:** A larger actor tests whether the current policy lacks capacity. Change only the actor dimensions; leave critic size and training length unchanged.
- **Training length:** Test more iterations only if checkpoint results are still improving near the end. Your two recent runs peaked near checkpoint 1000 and then got slower, so first check whether a longer run has evidence behind it.

## Use a simple experiment ledger

| Run | Hypothesis / one change | Best checkpoint | Score / finishes | Compare with baseline | Decision |
|---|---|---:|---|---|---|
| Baseline | `forward_reward = 2.5` | 1000 | 3.864 s / 5 of 5 | — | Keep as reference |
| Candidate | `forward_reward = 3.0` | pending | pending | pending | pending |

After each run:

1. Evaluate every checkpoint with the local Arena scorer.
2. Select the best checkpoint by full finishes first, then lowest mean time.
3. Compare against the baseline.
4. Repeat promising changes with another training seed.
5. Only consider publishing after the result is repeatable; the Arena entry confirms whether it actually improves your leaderboard position.

That gives each run a learning purpose: you’ll know whether a reward, training setting, or random seed explains the outcome.