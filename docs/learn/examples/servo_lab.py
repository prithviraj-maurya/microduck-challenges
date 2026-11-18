"""One XL330 servo on paper: how a position target becomes a torque.

No simulation here, only the servo's equations (the BAM model training uses).
Answers: how much torque does a 5 degree error produce? When does the servo
saturate? How much torque is lost when the joint is already moving?

    uv run --group workshop python docs/learn/examples/servo_lab.py
    uv run --group workshop python docs/learn/examples/servo_lab.py --vin 6.5
"""

import argparse
import math

import numpy as np
from arena.actuator import ERROR_GAIN, KP_FW, KT, MAX_CURRENT, R, VIN_NOMINAL, control_torque


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vin", type=float, default=VIN_NOMINAL, help="battery voltage (V); training samples 6.5 to 8.2")
    args = ap.parse_args()

    print(f"XL330, BAM m6 model: kt={KT:.3f} Nm/A  R={R:.2f} ohm  firmware kp={KP_FW:.0f}  vin={args.vin:.2f} V")
    print(f"current limit {MAX_CURRENT} A -> at most {KT * MAX_CURRENT:.2f} Nm while not moving")
    full_duty_err = 1.0 / (KP_FW * ERROR_GAIN)
    print(f"duty cycle reaches 100% at a position error of {math.degrees(full_duty_err):.1f} deg\n")

    print("Holding still (joint velocity 0): torque for a given position error")
    print(f"{'error (deg)':>12} {'torque (Nm)':>12}")
    for deg in (0.5, 1, 2, 5, 10, 20, 45):
        tau = control_torque(np.array([math.radians(deg)]), np.array([0.0]), np.array([0.0]), args.vin)[0]
        print(f"{deg:>12} {tau:>12.3f}")

    print("\nSame 10 deg error while the joint is already turning toward the target")
    print(f"{'speed (rad/s)':>14} {'torque (Nm)':>12}   back-EMF eats torque as the motor spins")
    for w in (0.0, 2.0, 5.0, 10.0, 15.0):
        tau = control_torque(np.array([math.radians(10)]), np.array([0.0]), np.array([w]), args.vin)[0]
        print(f"{w:>14} {tau:>12.3f}")

    no_load = args.vin / KT
    print(f"\nNo-load top speed at full duty: about {no_load:.1f} rad/s ({math.degrees(no_load) / 360 * 60:.0f} rpm)")


if __name__ == "__main__":
    main()
