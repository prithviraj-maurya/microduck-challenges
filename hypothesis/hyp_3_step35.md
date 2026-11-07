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
Experiment commit:
Training settings: step_reward = 3.5; forward_reward = 3.0; upright_reward = 2.0
Training seed: 2
HF checkpoint repo: prithviraj-maurya/mjlab-sprint2m-microduck-step35-seed2
HF job ID / actual cost:
Best checkpoint / command speed:
Local score / finished seeds:
Per-seed times or failures / replay seed:
Training curves:
Official Arena score / finishes / rank / policy URL: not submitted
Conclusion:
```

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
