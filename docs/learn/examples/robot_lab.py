"""Poke the duck in simulation, no policy involved.

Experiments (pick one with --mode):

  hold   hold the STAND pose with fixed targets, no policy. The soft servos sag
         and the duck tips forward within about 1.5 s: standing still is itself
         a balance skill the policy has to learn.
  joint  hang the duck in the air (trunk pinned), then move one joint by --deg
         and hold it; prints how far the joint really got and its torque.
  wave   hang the duck, swing one joint in a sine wave (--deg amplitude, --hz
         frequency); prints how far the joint lags its target.
  limp   switch the servos off; the duck collapses. Prints the tilt and the
         projected-gravity z the Arena uses to call a fall (> -0.5 for 0.3 s).

The walk model only gives the feet collision shapes, so a fallen duck sinks
through the floor: training ends the episode at a fall, so nothing else needs to
touch the ground.

Add --view to watch it in MuJoCo's viewer. On macOS the viewer needs `mjpython`
(see docs/SPRINT_2M_COMMAND_REFERENCE.md section 9); headless runs need nothing.

    uv run --group workshop python docs/learn/examples/robot_lab.py --mode hold
    uv run --group workshop python docs/learn/examples/robot_lab.py --mode joint --joint left_knee --deg 30
    uv run --group workshop python docs/learn/examples/robot_lab.py --mode wave --joint neck_pitch --deg 20 --hz 2
    uv run --group workshop python docs/learn/examples/robot_lab.py --mode limp
"""

import argparse
import math
import time

import mujoco
import numpy as np

from _sim import DEFAULT_POSE, JOINT_NAMES, control_step, load_duck, servos_for, trunk_tilt_deg

CTRL_DT = 0.02  # 50 Hz, as the policy runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("hold", "joint", "wave", "limp"), default="hold")
    ap.add_argument("--joint", choices=JOINT_NAMES, default="left_knee")
    ap.add_argument("--deg", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer (macOS: run with mjpython)")
    args = ap.parse_args()

    model, data = load_duck()
    servos, qpos_adr, dof_adr = servos_for(model)
    j = JOINT_NAMES.index(args.joint)
    trunk = model.body("trunk_base").id
    hang = args.mode in ("joint", "wave")
    if hang:
        data.qpos[2] += 0.05  # lift the feet off the floor
        mujoco.mj_forward(model, data)
    pin = data.qpos[:7].copy()
    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    steps = int(args.seconds / CTRL_DT)
    worst_lag = 0.0
    print(f"mode={args.mode}  joint={args.joint}  ({args.seconds:.1f} s at 50 Hz)")
    print(f"{'t (s)':>6} {'trunk z (m)':>11} {'tilt (deg)':>10} {'gravity z':>9} {'joint (deg)':>11} {'target':>7} {'torque (Nm)':>11}")
    for k in range(steps):
        t = k * CTRL_DT
        target = DEFAULT_POSE.copy()
        if args.mode == "joint" and t >= 0.5:
            target[j] += math.radians(args.deg)
        elif args.mode == "wave":
            target[j] += math.radians(args.deg) * math.sin(2 * math.pi * args.hz * t)

        if args.mode == "limp" and t >= 0.5:
            for _ in range(4):
                data.ctrl[:] = 0.0  # servos off: no torque at all
                mujoco.mj_step(model, data)
        elif hang:
            for _ in range(4):
                data.qpos[:7], data.qvel[:6] = pin, 0.0  # trunk held in the air, like a test stand
                servos.drive(data, target)
                mujoco.mj_step(model, data)
        else:
            control_step(model, data, servos, target)

        q = data.qpos[qpos_adr[j]]
        worst_lag = max(worst_lag, abs(q - target[j])) if t > 0.5 else worst_lag
        if k % 10 == 0 or k == steps - 1:
            g_z = data.xmat[trunk].reshape(3, 3)[2, 2] * -1.0  # projected gravity z: -1 upright, 0 on its side
            print(
                f"{t:>6.2f} {data.qpos[2]:>11.4f} {trunk_tilt_deg(model, data):>10.1f} {g_z:>9.2f} "
                f"{math.degrees(q):>11.1f} {math.degrees(target[j]):>7.1f} {data.actuator_force[j]:>11.3f}"
            )
        if viewer is not None:
            viewer.sync()
            time.sleep(CTRL_DT)
            if not viewer.is_running():
                break

    if args.mode == "joint":
        gap = math.degrees(data.qpos[qpos_adr[j]] - target[j])
        print(f"\n{args.joint} settled {gap:+.1f} deg from its target, holding {data.actuator_force[j]:+.3f} Nm")
    elif args.mode == "wave":
        print(f"\nlargest gap between {args.joint} and its moving target: {math.degrees(worst_lag):.1f} deg")


if __name__ == "__main__":
    main()
