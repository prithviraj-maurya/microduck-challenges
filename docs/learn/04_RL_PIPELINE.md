# 4. How RL is implemented

This guide follows one training run of `Mjlab-Sprint2m-MicroDuck` through the code: what one
environment step does, how rewards are computed, and what one PPO update does. It ends with how
to read the numbers in the training log. Every number here was printed from the live config on
2026-11-18 (your hypothesis 4 version of `env.py`).

## The problem, in RL words

| RL word | In the Sprint |
|---|---|
| **Agent / policy** | A neural network: 61 numbers in, 14 numbers out |
| **Environment** | One simulated duck on flat ground, 20 s per episode |
| **Observation** | The 61 numbers (below) |
| **Action** | 14 joint-angle offsets from the STAND pose, in radians |
| **Reward** | A weighted sum of 19 terms, computed every step |
| **Episode end** | Tilted more than 70° (`fell_over`), 20 s up (`time_out`), off the terrain, or a NaN in the physics |
| **Goal** | Maximize the discounted sum of future rewards |

Training runs **4096 copies** of the environment at once on one GPU. Each copy is a different
duck: different mass, friction, battery and delays (domain randomization).

## Timing: three clocks

| Clock | Period | Set by |
|---|---|---|
| Physics step | 5 ms (200 Hz) | `cfg.sim.mujoco.timestep = 0.005` |
| **Control step** (the policy acts) | 20 ms (50 Hz) = 4 physics steps | `decimation = 4` |
| PPO iteration | 24 control steps per duck = 98,304 transitions | `num_steps_per_env = 24` |

An episode is 20 s, which is 1000 control steps. Your 3000-iteration run collected
3000 × 98,304 ≈ **295 million** control steps. At 2.71 s per iteration on an L4 GPU, that is
about 36,000 simulated steps per second.

## One environment step

This is `ManagerBasedRlEnv.step()` in `mjlab/envs/manager_based_rl_env.py`, simplified:

```text
step(action):                                   # action: (4096, 14)
    action_manager.process_action(action)       # target = STAND + 1.0 × action
    repeat 4 times:                             # decimation
        action_manager.apply_action()           # write the joint targets
        scene.write_data_to_sim()               # the BAM actuator turns targets into torques
        sim.step()                              # MuJoCo Warp: 5 ms of physics, all 4096 ducks
    terminated = termination_manager.compute()  # fell over? time out?
    reward     = reward_manager.compute(dt)     # Σ weight × term × 0.02 s
    reset the ducks that ended                  # event_manager "reset" mode: new random duck
    sim.forward()
    command_manager.compute()                   # maybe resample the speed command
    event_manager.apply("interval")             # maybe shove a duck
    obs = observation_manager.compute()         # the next 61 numbers (+ noise, delays)
    return obs, reward, terminated, time_outs
```

Everything you configure in `env.py` is a **term** in one of these managers. Each manager reads
a dict of terms (`cfg.rewards[...]`, `cfg.observations[...]`, `cfg.events[...]`, …), calls each
term's function on all 4096 ducks at once, and combines the results. To change the task, you add,
remove or re-weight terms.

## Observations: what the policy sees (61)

| Slice | Term | Size | Noise and delay in training |
|---|---|---:|---|
| 0–2 | `base_ang_vel`: trunk rotation rate (gyro) | 3 | ±0.03 rad/s, 0–1 step delay, up to 6° IMU mounting error |
| 3–5 | `projected_gravity`: "down", seen from the trunk | 3 | ±0.01, 0–1 step delay, same IMU error |
| 6–19 | `joint_pos`: angles relative to STAND | 14 | ±0.001 rad, plus a fixed per-robot encoder offset of up to ±0.86° |
| 20–33 | `joint_vel`: joint speeds | 14 | ±0.25 rad/s, always one step old |
| 34–47 | `actions`: what it did last step | 14 | |
| 48–50 | `command`: speed order (vx, vy, yaw rate) | 3 | vx 0.20–1.20 m/s, vy and yaw rate always 0 |
| 51–54 | `head_command`: head pose order | 4 | small random offsets (±0.05 rad) |
| 55–60 | `body_command`: body pose order | 6 | tiny; this slot is unused in the Sprint but kept alive |

Three design points:

- **No position, no speed, no heading.** The actor never learns where it is on the lane or how
  fast it is going. It has to infer speed from how its body moves. The real robot has no
  sensor for these either.
- **The command slots are shared by every Microduck policy** (walk, stand, roulade, …). That
  is what lets the robot hot-swap policies, and why unused slots are zero-padded, never
  removed.
- **The noise and delays are deliberate.** The real IMU and encoders are noisy and late. A
  policy trained on perfect data will fail on the robot.

**The critic sees more (77 numbers).** PPO trains two networks: the *actor* (the policy) and
the *critic*, which estimates how much future reward a state is worth. The critic only exists
during training, so it can be given information the robot will never have. This is called
"privileged information" or "asymmetric actor-critic". The critic gets:

- the true trunk velocity;
- the feet's heights, air times, contact states and contact forces;
- the heading error;
- all of it without noise.

Better value estimates mean better advantage estimates, which mean better updates for the
actor.

## Actions: what the policy controls (14)

`JointPositionAction` with `scale = 1.0` and `use_default_offset = True`:

```text
joint target (rad) = STAND pose + 1.0 × action
```

So `action = 0` means "hold STAND". An action of 0.3 on the knee asks for 17° more bend than
STAND. The BAM actuator then decides how much torque that target produces (guide 1). Actions are
not clipped and not filtered: a policy is free to ask for anything, and the soft servos and the
`action_rate_l2` penalty keep it in check.

## Rewards: what the policy is paid for (19 terms)

Each step, `reward = Σ weight × term(state) × dt` (dt = 0.02 s). Positive weights pay, negative
weights charge.

| Term | Weight | What it measures |
|---|---:|---|
| `lane_progress` | **+10** | Speed down the lane ÷ 2 m/s, capped at 1. *Yours (hyp 4)* |
| `track_linear_velocity` | +2 (`forward_reward`) | Bell curve on speed error; slower than the command costs, faster is free. *Yours* |
| `track_angular_velocity` | +2 | Bell curve on yaw-rate error (the command is always 0) |
| `upright` | +2 (`upright_reward`) | Trunk upright, with a 12° forward lean target at speed. *Yours* |
| `head_pose_tracking` | +2 | Head joints match the head command; carried forward at speed. *Yours* |
| `lane_drift` | +2 | Bell curve on sideways speed across the lane. *Yours* |
| `air_time` | +3 (`step_reward`) | Pays a foot for 0.125–0.3 s swing phases: real steps, not shuffling |
| `pose` | +1 | Joints near STAND, with wider tolerance when walking than when standing |
| `heading_hold` | +1 | Trunk keeps the heading it spawned with |
| `foot_clearance` | −2 | Swinging foot below 2 cm |
| `foot_swing_height` | −0.25 | Peak swing height off 2 cm |
| `foot_slip` | −0.1 | Feet sliding while in contact |
| `action_rate_l2` | −0.1 → −0.3 (curriculum) | Squared change in action between steps: jerkiness |
| `body_ang_vel` | −0.05 | Trunk rotation rate |
| `angular_momentum` | −0.02 | Whole-body spin |
| `dof_pos_limits` | −1 | Joints near their limits |
| `self_collisions` | −1 | Body parts hitting each other |
| `body_pose_tracking`, `head_pose_bias` | 0 | Kept so their slots stay alive for other tasks |

The function behind each name is in one of three places: `sprint_2m/rewards.py` (yours),
`mjlab_microduck/tasks/mdp.py`, or `mjlab/tasks/velocity/mdp/rewards.py`.

### Curricula: the schedule

A curriculum changes the task as training goes on. Steps are counted as iterations × 24.

| Curriculum | What it changes |
|---|---|
| `action_rate_weight` | `action_rate_l2` −0.1 → −0.2 at iteration 500 → −0.3 at 750 (was up to −1.0 before hyp 4) |
| `standing_envs` | Share of ducks told to stand still: 2% (was ramped to 25%) |
| `com_range`, `head_com_range` | Widen the centre-of-mass randomization (±3 mm at the start) |
| `body_pose_range`, `head_pose_bias_weight` | For the unused slots |

Your hypotheses 1–3 showed how much a curriculum can matter: the `action_rate_weight` ramp alone
explained why checkpoint 1000 always won.

### Events: randomization and shoves

| When | Events |
|---|---|
| Once, at startup | Foot friction 0.7–1.3, encoder offsets ±0.015 rad, mass and inertia ±5%, battery 6.5–8.2 V and voltage sag (inside the actuator) |
| At every episode reset | Spawn pose, joint noise, trunk and head centre of mass (±3 mm, widened by curriculum), servo friction ±10%, rotor inertia ±10% |
| Every 3–6 s | `push_robot`: an instant ±0.3 m/s kick |

Each of the 4096 ducks is a slightly different robot. A policy that wins on all of them is likely
to work on the real one. This is the core sim-to-real idea (domain randomization). The shoves are
why `fell_over` stays high in training even when the Arena, which has no shoves, sees no falls.

## PPO: how the network learns

The loop is `OnPolicyRunner.learn()` in `rsl_rl/runners/on_policy_runner.py`. Each iteration:

**1. Collect (rollout).** For 24 steps, every duck acts. The actor outputs a *mean* action; a
Gaussian with a learned standard deviation (starting at 1.0) adds exploration noise. The runner
stores observations, actions, rewards, the critic's value estimates and the action
probabilities.

**2. Compute advantages (GAE).** `PPO.compute_returns()` works backwards through the 24 steps:

```text
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)            γ = 0.99
A_t = δ_t + γ·λ·A_{t+1}                       λ = 0.95
```

- **Advantage** = "this action turned out better (or worse) than the critic expected".
- **γ = 0.99** sets how far ahead the policy plans. Rewards about 100 steps (2 s) away still
  count for about a third. For a 1.2 s sprint, that horizon covers the whole race.
- **λ = 0.95** trades bias against noise in the estimate.

Advantages are then normalized to mean 0, standard deviation 1.

**3. Update (5 epochs × 4 mini-batches = 20 gradient steps).** `PPO.update()` minimizes:

```text
loss = surrogate + 1.0 × value_loss − 0.01 × entropy + 0.5 × mirror_loss
```

| Part | What it does | Setting |
|---|---|---|
| **Clipped surrogate** | Raise the probability of actions with positive advantage, lower the others. The ratio new/old probability is clipped to [0.8, 1.2], so one update can't move the policy far. This clip is the "proximal" in PPO | `clip_param = 0.2` |
| **Value loss** | Train the critic to predict the returns (also clipped) | `value_loss_coef = 1.0` |
| **Entropy bonus** | Pays for keeping some randomness, so exploration doesn't stop too early | `entropy_coef = 0.01` |
| **Mirror loss** (hyp 4) | Mirror the observation left↔right, ask the actor again, and penalize it if the answer isn't the mirrored action. Pushes towards symmetric gaits | `mirror_loss_coeff = 0.5` |
| **Adaptive learning rate** | After each mini-batch, measure how far the policy moved (KL divergence). If above 0.02, divide the learning rate by 1.5; if below 0.005, multiply by 1.5 | starts at 1e-3, `desired_kl = 0.01` |
| **Gradient clipping** | Caps the size of an update | `max_grad_norm = 1.0` |

**4. Log and save.** Losses, rewards and metrics go to TensorBoard. A checkpoint (`model_N.pt`)
is saved every 100 iterations (`save_interval`); with `--hf-jobs` it is uploaded to the Hub.

### The networks

| | Actor (policy) | Critic (value) |
|---|---|---|
| Input | 61 | 77 |
| Hidden layers | 512 → 256 → 128, ELU | 512 → 256 → 128, ELU |
| Output | 14 action means (+ 1 learned noise level) | 1 value |
| Input normalization | running mean/std, learned from data | same |

The **observation normalizer** subtracts a running mean and divides by a running standard
deviation for each of the 61 inputs, so a joint velocity of 10 rad/s and a gravity component of
0.1 reach the network on similar scales. It is part of the policy. That is why export must bake it
into the ONNX (guide 5).

## Reading the training log

| Line | Meaning | Healthy |
|---|---|---|
| `Mean reward` | Average total reward per episode | Rising |
| `Mean episode length` | Steps before an episode ended (max 1000) | High; drops mean falls |
| `Episode_Reward/<term>` | Average per-episode sum of that weighted term **÷ 20 s** | Positive terms rising, penalties ≤ 0 |
| `Episode_Termination/fell_over` | Share of ended episodes that ended in a fall | Lower is better, but shoves keep it high |
| `Curriculum/*` | Where each schedule is | Check it whenever another curve moves suddenly |
| `Metrics/lane_speed_mean` | **Your new metric:** m/s down the lane | The one that tracks the race time |
| `Mean action std` (TensorBoard: `Policy/mean_std`) | The exploration noise | Slowly falling |
| `Loss/learning_rate` | The adaptive learning rate | Moves; a collapse to 1e-5 means updates are too violent |
| `Mean value loss`, `Mean surrogate loss`, `Mean entropy` | The PPO losses | Value loss should not explode |
| `Mean symmetry loss` | Mirror loss (hyp 4) | Small, about 0.01–0.02 |

**Why `Episode_Reward` numbers look small.** mjlab divides each episode's sum by the full
episode length (20 s), even when the episode ended early. A term with weight 2 that paid its
maximum every step would read 2.0 only if the episode lasted the full 20 s. Episodes that end in
falls or resets read lower. This is one reason your `track_linear_velocity` sat around 0.5: the
reading mixes how well the term was met with how long episodes lasted.

The rule your hypotheses kept proving: **reward curves show what the policy collects, not how fast
it is.** Only `arena score` judges a policy.

Next: [5. From policy to motors](05_POLICY_TO_MOTORS.md).
