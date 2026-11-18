# 2. Try it in simulation

Everything in this guide runs on your Mac's CPU and costs nothing. It goes from the smallest
piece (one servo) to the whole thing (a policy racing in the Arena).

All commands run from the repo root. They need the `workshop` dependency group, which installs
the Arena package (the examples borrow its servo model):

```bash
export UV_CACHE_DIR=/private/tmp/microduck-uv-cache UV_LINK_MODE=copy   # as in the command reference
```

## Which tool for which question

| Question | Tool | Time |
|---|---|---|
| How does one servo turn an error into torque? | `examples/servo_lab.py` | instant |
| How do the joints behave, without a policy? | `examples/robot_lab.py` | seconds |
| What exactly goes in and out of the policy each step? | `examples/policy_loop.py` | seconds |
| **How good is this policy?** (the only number that counts) | `arena score` | about 7 s |
| What does a run look like? Replay a seed | `workshop sprint-2m` (browser) | interactive |
| Drive a policy with the keyboard in a 3D viewer | `infer` / `scripts/infer_policy.py` under `mjpython` | interactive |

## Step 1: one servo, on paper

```bash
uv run --group workshop python docs/learn/examples/servo_lab.py
uv run --group workshop python docs/learn/examples/servo_lab.py --vin 6.5   # a nearly flat battery
```

It prints the servo's torque for a range of position errors, then the torque for one error as
the joint speeds up. What to notice:

- 0.010 Nm per degree of error: the servos are soft;
- the torque collapses as the joint turns (back-EMF);
- a lower battery voltage makes everything weaker.

That last point is why training randomizes the battery between 6.5 and 8.2 V.

## Step 2: poke the robot

[examples/robot_lab.py](examples/robot_lab.py) loads the same robot model training uses, with
its 14 servos driven by the BAM servo model. No policy is involved: you set the targets.

```bash
E=docs/learn/examples
uv run --group workshop python $E/robot_lab.py --mode hold
uv run --group workshop python $E/robot_lab.py --mode joint --joint left_knee --deg 30
uv run --group workshop python $E/robot_lab.py --mode wave  --joint left_knee --deg 20 --hz 2
uv run --group workshop python $E/robot_lab.py --mode limp
```

What each one shows. These are measured results:

**`hold`: standing still is a skill.** All 14 targets are held at the STAND pose, with no
feedback.

- The knees sag about 10° under the trunk's weight.
- The trunk tips forward: 10° at 0.6 s, 25° at 1.0 s.
- The duck is down by 1.4 s.

MuJoCo's own ideal position servos give the same result. The STAND pose isn't self-balancing;
a policy has to keep catching it. This is why `upright`, `pose` and the standing environments
exist in the Sprint's rewards.

**`joint`: soft servos stop short.** With the duck hung in the air, a knee told to move +30°
settles 1.1° short. It holds 0.011 Nm there, about the servo's static friction: below that,
nothing moves.

**`wave`: servos lag.** A knee swinging ±20° falls behind its target by:

| Swing frequency | Largest gap |
|---:|---:|
| 0.5 Hz | 8.6° |
| 1 Hz | 15.3° |
| 2 Hz | 23° |
| 4 Hz | 24.6° |

A sprinting gait steps at a few hertz, so the policy has to command where it wants the leg to
*be*, well ahead of time. It can't correct a step after it starts.

**`limp`: what a fall looks like to the Arena.** With the servos off, the duck collapses within
a second.

- Watch the `gravity z` column go from −1.00 to above −0.5. Held there for 0.3 s, that is
  the Arena's definition of a fall.
- The duck then sinks through the floor. That is expected: the walk model only lets the feet
  touch the ground (guide 1).

Add `--view` to watch any of these in MuJoCo's 3D viewer. On macOS the viewer needs `mjpython`
instead of `python`; section 9 of the [command reference](../SPRINT_2M_COMMAND_REFERENCE.md)
shows how you set that up in `microduck_rl`. Headless runs need nothing extra.

Things to try:

- Swing `neck_pitch` in `wave` mode. The head is heavy, so watch how much worse it tracks
  than a knee.
- Edit `_sim.py` to change `KP_FW` and see how much stiffer the joints get.

## Step 3: the whole control loop, in 60 lines

[examples/policy_loop.py](examples/policy_loop.py) runs an exported policy the way the Arena
does: sensors → 61 numbers → the network → 14 targets → servos → physics, 50 times a second.
Read it first; it is the shortest complete description of what a policy *is*.

```bash
ONNX=publish-evals/publish-lx-seed2-cp2999/policy.onnx      # your published 1.229 s policy
uv run --group workshop python docs/learn/examples/policy_loop.py $ONNX --vx 0.85 --seconds 3 --print-obs
```

Measured output, shortened:

```text
     command [0.85 0. 0. ...]
t= 0.5s  x= 0.258 m  tilt=  8.8 deg
t= 1.0s  x= 0.947 m  tilt= 12.7 deg
t= 1.5s  x= 1.793 m  tilt= 12.3 deg
t= 2.0s  x= 2.682 m  tilt= 12.7 deg
travelled 4.372 m forward in 3.0 s: average 1.46 m/s (commanded 0.85)
```

Two things in there connect straight back to your hypothesis 4:

- **The duck ignores the command.** Told 0.85 m/s, it covers 1.74 m in the second
  second (1.0 to 2.0 s): about 1.7 m/s once up to speed. That is the effect of paying for lane speed directly (`lane_progress`) and not
  charging for going faster than told.
- **It runs with the trunk tilted about 12° forward.** That is exactly the 12° lean target in
  `upright_lean`. The reward asked for a forward lean at speed, and the policy took it.

This loop leaves out a few things the Arena does (lane walls, a 0.5 s settle before the clock
starts, start noise, one extra servo update per step), so its times won't match `arena score`.
Use it to understand and experiment, not to judge.

Experiments worth trying:

- Change `command[0]` and watch the speed *not* change.
- Set the head command (`command[3:7]`) and see whether the policy follows it.
- Add noise to `obs[3:6]` (a badly mounted IMU) and see how much it takes to make the duck
  fall.

## Step 4: judge a policy, the way the leaderboard does

```bash
uv run --group workshop arena score publish-evals/publish-lx-seed2-cp2999/policy.onnx --event sprint-2m
```

This is the real evaluator, the same code the Arena runs:

1. It races the policy at 21 commanded speeds from 0.20 to 1.20 m/s, on seed 1.
2. It reruns the 3 fastest on seeds 1–5.
3. It reports the best mean time.

A fall (tilted over 60° for 0.3 s) or no finish in 50 s counts as 50 s. Section 7 of the
command reference shows how to export a checkpoint to ONNX first.

Only this number says whether a policy got better. Training curves are clues (guide 4).

## Step 5: watch it

- **`uv run --group workshop workshop sprint-2m`** opens the workshop in your browser. You can
  pick a run, a checkpoint and a seed, and replay the evaluator's own run. It is the closest
  thing to watching the official race.
- **`infer` under `mjpython`** opens MuJoCo's viewer and lets you drive a policy with the
  keyboard (the terminal prints the keys). Section 9 of the
  command reference has the exact commands. It is a rehearsal of the real robot's runtime, not
  the Arena lane, so it has no seeds and no timing.
- **mjlab's `play` command with `--viewer viser`** (a browser viewer) does **not** work for the
  Sprint task in this version. It starts, then crashes with
  `add_slider: initial_value (0.0) is outside of [min, max] = [0.1, 10.0]` in mjlab's velocity
  command panel. Tested on 2026-11-18; use the two tools above instead.

## Step 6: train, for pennies, before training for dollars

A smoke test checks that the whole training stack builds and steps. You did this before every
paid run:

```bash
uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 2 --agent.max-iterations 2
```

It runs on the CPU. It won't learn anything at this size, but it catches about every config
error: a reward that crashes, a wrong observation size, a typo in a weight. Real training needs
a CUDA GPU, which is what `--hf-jobs` rents.

Next: [3. Code tour](03_CODE_TOUR.md).
