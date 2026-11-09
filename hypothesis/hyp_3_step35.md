# Hypothesis 3: raise the foot-lift reward to 3.5

## Hypothesis

**Hypothesis:** Raising `step_reward` from **3.0 to 3.5** will encourage stronger foot swings, helping the duck move forward faster while preserving all five finishes.

This is an exploratory test. We have not yet confirmed from a replay that checkpoint 1000 drags or shuffles its feet, so the proposed mechanism is uncertain. The prior experiment notes list this as the next existing-knob candidate after testing the forward-tracking reward.

## One change

Change only `params.step_reward.value` in `src/microduck_challenges/sprint_2m/challenge.toml`:

```toml
[params.step_reward]
value = 3.5
```

Keep the other reward weights, environment, command range, PPO settings, training seed, and run length fixed. In particular, keep `forward_reward = 3.0`, `upright_reward = 2.0`, the full Arena command sweep of 0.20–1.20 m/s, and `heading_hold` unchanged.

## Baseline and decision rule

- **Baseline:** `step_reward = 3.0`, `forward_reward = 3.0`, training seed 2, checkpoint 1000: **3.560 s, 5/5 finishes**, at **0.70 m/s** in the local Arena evaluator.
- **Promising:** at least one checkpoint scores **3.460 s or faster with 5/5 finishes** in the same local evaluator. Score every checkpoint; the latest checkpoint is not assumed to be best.
- **Reject or leave inconclusive:** no 5/5 checkpoint reaches 3.460 s, or any apparent speedup comes with fewer finishes. Training reward alone is not evidence of a faster sprint.

The 0.10 s margin is a practical signal threshold, not a statistical guarantee. A promising result should be checked on another training seed before relying on it. The Arena's official score is the final leaderboard measure.

## Run record

```text
Experiment commit: 9e65b2a1a (clean provenance)
Training settings: step_reward = 3.5; forward_reward = 3.0; upright_reward = 2.0
Training seed: 2
HF checkpoint repo: prithviraj-maurya/mjlab-sprint2m-microduck-step35-seed2
HF job ID / actual cost: not recorded
Best checkpoint / command speed: 1000 / 0.70 m/s
Local score / finished seeds: 3.604 s / 5 of 5
Per-seed times or failures / replay seed: 3.58, 3.60, 3.60, 3.64, 3.60 s; replay seed 3
Training curves: at 1000, mean episode length 977.63/1000, fell_over 0.2500,
  weighted track_linear_velocity reward 1.2530, weighted air_time reward 1.4913.
  At 2999: 960.09/1000, 0.2917, 1.4464, and 1.2183 respectively.
Official Arena score / finishes / rank / policy URL: not submitted
Conclusion: rejected locally. Best score was 0.044 s slower than the 3.560 s
  step_reward=3.0 baseline and did not meet the 3.460 s decision threshold.
```

## Checkpoint sweep and what it showed

| Checkpoint | Local score | Finishes | Winning command |
|---:|---:|---:|---:|
| 0 | 50.000 s | 0/5 | 0.95 m/s |
| 250 | 8.976 s | 5/5 | 0.85 m/s |
| 500 | 8.072 s | 5/5 | 0.95 m/s |
| 750 | 31.620 s | 2/5 | 0.80 m/s |
| 1000 | **3.604 s** | **5/5** | **0.70 m/s** |
| 1250 | 3.784 s | 5/5 | 0.70 m/s |
| 1500 | 3.816 s | 5/5 | 0.70 m/s |
| 1750 | 3.864 s | 5/5 | 0.70 m/s |
| 2000 | 3.920 s | 5/5 | 0.70 m/s |
| 2250 | 4.008 s | 5/5 | 0.70 m/s |
| 2500 | 4.096 s | 5/5 | 0.70 m/s |
| 2750 | 3.908 s | 5/5 | 0.70 m/s |
| 2999 | 3.984 s | 5/5 | 0.70 m/s |

The higher foot-lift reward did not produce a faster checkpoint. Training reward
for `air_time` peaked at checkpoint 1000 and then fell; meanwhile the velocity
tracking reward continued rising, but evaluator time got worse. The best checkpoint
was still 1000, matching the earlier run's best-checkpoint timing pattern. The
curves explain training behavior but do not change the evaluator result.

## What was checked and how

1. Downloaded the completed job's checkpoints, saved configuration, event log, and
   provenance into `logs/rsl_rl/`.
2. Confirmed the run used seed 2, 4,096 environments, 3,000 iterations, commit
   `9e65b2a1a`, and `dirty: false`. The saved config has `air_time` weight 3.5 and
   velocity tracking weight 3.0.
3. Exported and scored all 13 available checkpoints with the local Sprint evaluator.
   Each score swept the Arena's 21 commands and checked the selected speed on five
   seeds. The scorer selected 0.70 m/s for the best checkpoint, with 5/5 finishes.
4. Read TensorBoard values for episode length, falls, weighted velocity reward, and
   weighted air-time reward at checkpoints 1000 and 2999. These were diagnostics,
   not the success metric.

The local score is not an official Arena result. No policy was published or entered.

## Budget and launch command

The Sprint guide estimates 1–2.5 hours for a 3,000-iteration `l4x1` run at $0.80/hour (about $0.80–$2.00 if it runs for that full interval; check the actual job charge). This command starts a paid Hugging Face Job. Run it only when ready to spend that amount:

```bash
uv run train Mjlab-Sprint2m-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.seed 2 \
  --agent.logger tensorboard \
  --agent.run-name step35-seed2 \
  --hf-jobs --no-wandb --timeout 3h \
  --namespace prithviraj-maurya \
  --ckpt-repo prithviraj-maurya/mjlab-sprint2m-microduck-step35-seed2
```

After the run, download and score each checkpoint with the local Arena evaluator. Do not publish or enter a policy as part of this experiment.
