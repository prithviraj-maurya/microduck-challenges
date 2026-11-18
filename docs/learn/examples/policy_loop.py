"""The whole control loop in about 60 lines: sensors -> 61 numbers -> ONNX -> 14 targets -> servos.

This is a simplified copy of what the Arena does (arena/runner.py): same robot,
same servo model, same observation layout, same 50 Hz loop. It leaves out the
lane walls, the start settle, the start noise and the extra servo drive the
Arena runs after each step, so its times will not match `arena score` exactly.
Use it to read and change the loop, and `arena score` to judge a policy.

    uv run --group workshop python docs/learn/examples/policy_loop.py publish-evals/<export>/policy.onnx --vx 0.85
"""

import argparse

import mujoco
import numpy as np
import onnxruntime as ort

from _sim import DEFAULT_POSE, control_step, load_duck, servos_for, trunk_tilt_deg

CTRL_DT = 0.02


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--vx", type=float, default=0.5, help="commanded forward speed (m/s)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--print-obs", action="store_true", help="print the first observation, labelled")
    args = ap.parse_args()

    session = ort.InferenceSession(args.onnx)
    inp, out = session.get_inputs()[0].name, session.get_outputs()[0].name
    model, data = load_duck()
    servos, qpos_adr, dof_adr = servos_for(model)
    gyro = model.sensor("imu_ang_vel").adr[0]
    trunk = model.body("trunk_base").id

    command = np.zeros(13, dtype=np.float32)  # [vx, vy, yaw rate | head 4 | body 6]
    command[0] = args.vx
    last_action = np.zeros(14, dtype=np.float32)
    prev_qvel = data.qvel[dof_adr].copy()  # the joint-velocity input is one step old, as in training

    for k in range(int(args.seconds / CTRL_DT)):
        # 1. Sensors -> the 61 numbers the policy sees.
        obs = np.empty(61, dtype=np.float32)
        obs[0:3] = data.sensordata[gyro : gyro + 3]                              # trunk angular velocity (rad/s)
        obs[3:6] = data.xmat[trunk].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])  # gravity seen from the trunk
        obs[6:20] = data.qpos[qpos_adr] - DEFAULT_POSE                           # joint angles, relative to STAND
        obs[20:34] = prev_qvel                                                   # joint velocities
        obs[34:48] = last_action                                                 # what it did last step
        obs[48:61] = command                                                     # what it is asked to do
        if args.print_obs and k == 0:
            for name, a, b in (("ang vel", 0, 3), ("gravity", 3, 6), ("joint pos", 6, 20), ("joint vel", 20, 34),
                               ("last action", 34, 48), ("command", 48, 61)):
                print(f"{name:>12} {np.round(obs[a:b], 3)}")
        prev_qvel = data.qvel[dof_adr].copy()

        # 2. The network: 61 numbers in, 14 numbers out (normalizer baked into the ONNX).
        last_action = session.run([out], {inp: obs.reshape(1, 61)})[0].reshape(14).astype(np.float32)

        # 3. Action -> joint targets: an offset from the STAND pose, in radians.
        target = DEFAULT_POSE + last_action * 1.0

        # 4. Servos chase the targets for 4 physics steps of 5 ms (BAM XL330 model).
        control_step(model, data, servos, target)

        if k % 25 == 0:
            print(f"t={k * CTRL_DT:4.1f}s  x={data.qpos[0]:6.3f} m  y={data.qpos[1]:6.3f} m  tilt={trunk_tilt_deg(model, data):5.1f} deg")
        if trunk_tilt_deg(model, data) > 60:
            print(f"fell at t={k * CTRL_DT:.2f}s")
            return

    t = args.seconds
    print(f"\ntravelled {data.qpos[0]:.3f} m forward in {t:.1f} s: average {data.qpos[0] / t:.2f} m/s (commanded {args.vx})")


if __name__ == "__main__":
    main()
