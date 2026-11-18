# 1. The robot

Microduck is a small two-legged robot by Pollen Robotics: about 25 cm tall and about 800 g on
the real robot. In simulation it weighs 737 g (summed from the model file). It walks on two
five-joint legs and carries a heavy head on a four-joint neck. Everything the policy does, it
does by sending 14 target angles to 14 servos, 50 times a second.

## The body, by mass

These numbers come from the simulation model `robot_walk.xml`. The CAD was exported from Onshape
with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot).

| Part | Mass | Share |
|---|---:|---:|
| Trunk (frame, electronics, battery) | 199 g | 27% |
| Head assembly (neck + head + jaw) | 280 g | 38% |
| Each leg (hip yaw link down to the foot) | 129 g | 17.5% each |
| **Total** | **737 g** | |

**The head is 38% of the robot.** That one fact explains a lot of the Sprint work. A heavy head
swinging on a neck is a large moving counterweight. Your hypothesis 4 recipe pulls the head
forward when running for that reason. It is also why `head_pose_tracking` is one of the main
rewards, not a detail.

What the CAD contains, from the part names in `mjlab_microduck/robot/microduck/assets/`:

- 14 `xl330` servo housings;
- foot soles;
- trunk shells and a `motor_support`;
- a `power_support`;
- a camera lens on an M12 lens holder;
- a speaker;
- a `pcb__raspberry_pi_zero_2_w` board and an `elec_rpi_robot_hat_pcb`;
- a part called `np_f970`, which I could not identify from the code.

The robot project's README says the current robot runs on a **Rockchip RK3566** board. Its design
doc mentions a "Zero 3W" board, likely the Radxa Zero 3W, which uses that chip; I have not
confirmed this. So the compute board in the simulation CAD (a Pi Zero 2 W) is probably older than
the shipping robot. It doesn't matter for the simulation: the board is only mass in the trunk.

## The 14 joints

The order below is the order of the 14 actions the policy outputs, the 14 joint values it
observes, and the servos on the bus. It never changes; every tool in these repos depends on it.

| # | Joint | Range | STAND pose | Moves |
|---:|---|---|---:|---|
| 0 | `left_hip_yaw` | −25° … +30° | 0° | turns the leg about the vertical |
| 1 | `left_hip_roll` | −22° … +22° | −5° | swings the leg sideways |
| 2 | `left_hip_pitch` | −90° … +90° | −26.2° | swings the leg forward and back |
| 3 | `left_knee` | −90° … +90° | −0.3° | bends the knee |
| 4 | `left_ankle` | −90° … +90° | +26.0° | tilts the foot |
| 5 | `neck_pitch` | −90° … +60° | +20° | tips the neck forward and back |
| 6 | `head_pitch` | −90° … +90° | +20° | nods the head |
| 7 | `head_yaw` | −170° … +170° | 0° | turns the head |
| 8 | `head_roll` | −25° … +25° | 0° | tilts the head sideways |
| 9–13 | `right_hip_yaw` … `right_ankle` | mirror of 0–4 | mirror of 0–4 | |

The STAND pose (`HOME_FRAME` in `robot/microduck_constants.py`, keyframe `STAND` in
`scene_walk.xml`) is the reference everything is measured from:

- the policy observes joint angles **relative to STAND**;
- its action is an **offset from STAND**: an action of all zeros means "hold STAND".

The pose has a history. It was shifted 5 mm forward so the centre of mass sits over the ankles.
Before that, the robot leaned back and the standing policy learned to droop its head as a
counterweight.

**Left and right use opposite signs** for the pitch joints (left hip pitch −26°, right +26°).
This is why the mirror-symmetry code (`tasks/symmetry.py`) negates those joints when it swaps
the legs.

The real robot has a **15th servo**, the mouth (jaw). The policy never sees or drives it. The
runtime leaves its slot at zero; see guide 5.

## The servos: Dynamixel XL330

The joints are Dynamixel XL330 servos: a small DC motor, a gearbox, a magnetic encoder on the
output, and a microcontroller running a position loop. The policy doesn't send torque. It sends
a **goal position**, and the servo's firmware decides how hard to push.

How hard? Training models this with **BAM** ("Better Actuator Models",
[Rhoban/bam](https://github.com/Rhoban/bam)). BAM was fitted to measurements of a real XL330 on a
test bench. The equations run in two steps:

```text
1. Firmware:  duty  = clip( kp × error_gain × (target − angle),  current window,  ±1 )
              voltage = duty × battery_voltage
2. Motor:     torque = kt × voltage / R  −  kt² × speed / R        (the second term is back-EMF)
   then friction (static, Stribeck and load-dependent) is applied by MuJoCo's solver.
```

| Constant | Value | Where it comes from |
|---|---|---|
| `kp` (firmware stiffness) | 200 | `_BAM_ACTUATOR_KWARGS` in `microduck_constants.py`; BAM's default is 400 |
| `kt` (torque constant) | 0.366 Nm/A | fitted, `bam/params/xl330/m6.json` |
| `R` (winding resistance) | 2.81 Ω | fitted, same file |
| Current limit | 1.75 A | BAM's XL330 firmware model |
| Battery voltage | 6.5–8.2 V, random per robot in training | `vin_range` |
| Voltage sag under load | 0–0.2 V per Nm of total torque | `vin_drop_gain_range` |
| Command delay | 3–6 physics steps (15–30 ms), random | `delay_min_lag`, `delay_max_lag` |

What those numbers mean in practice. These were measured with
[examples/servo_lab.py](examples/servo_lab.py) at 7.35 V:

| Position error | Torque while holding still |
|---:|---:|
| 1° | 0.010 Nm |
| 5° | 0.048 Nm |
| 20° | 0.19 Nm |
| 45° | 0.43 Nm |

- **The servos are soft:** about 0.01 Nm per degree of error. A joint holding the duck's
  weight has to be a few degrees "behind" its target to produce any torque. Run
  `robot_lab.py --mode hold` and watch the knees sag about 10° under the trunk.
- **Speed eats torque.** At 10° of error the servo gives 0.096 Nm standing still. Turning
  at 2 rad/s toward the target, it gives almost nothing, because back-EMF cancels the drive.
  No-load top speed is about 20 rad/s.
- **The servos are slow to respond.** Commands arrive 15–30 ms late, and the motor is soft.
  Measured with the duck hung in the air (`robot_lab.py --mode wave`), a knee told to swing
  ±20° lags its target by 9° at 0.5 Hz and by 23° at 2 Hz and above. A fast gait has to be
  planned ahead; it cannot be corrected after the fact.

The researchers modelled the servo this carefully because, as `microduck_rl`'s README puts it,
"at this scale — tiny servos driving a ~800 g biped — actuator fidelity is most of the sim2real
gap". An ideal servo in simulation would teach the policy moves the real one cannot make.

**Backlash.** The real gearboxes have about ±1° of play. Every main task has a `-Backlash-`
variant that adds a free ±1° hinge after each servo, and reads the encoder through it, as the
real encoder does. Your Sprint task uses the plain model.

## What the robot senses

| Sensor | In the policy's input? | Notes |
|---|---|---|
| IMU in the trunk (site `imu`) | **Yes**: angular velocity (3) and the direction of gravity (3) | On the real robot this is a separate IMU board on the servo bus. The runtime turns its quaternion into "gravity seen from the trunk" |
| Joint encoders (14) | **Yes**: angle (14) and velocity (14) | Read from each servo |
| Feet touching the ground | No; the critic only, in training | Simulated contact sensors. The real robot has no foot sensors |
| Head IMU, camera, ToF depth sensor, mouth tip | No | Sites exist in the model for other tasks and the runtime's other services |
| Speaker, gamepad, NFC, Bluetooth, wifi | No | Runtime features (guide 5) |

**Projected gravity** is worth understanding, because the Arena uses it to call a fall. It is
the world's "down" direction expressed in the trunk's own frame:

- standing straight, it reads `(0, 0, −1)`;
- leaning nose-down, the first number becomes positive;
- lying on its side, the last number reaches 0.

The Arena calls a fall when the last number stays above −0.5 (a tilt over 60°) for 0.3 s.

## Simulation models of the same robot

`mjlab_microduck/robot/microduck/` holds several versions of one robot. They differ in which
parts can touch things:

| Model | Collision shapes | Used by |
|---|---|---|
| `robot_walk.xml` | feet only | Velocity tasks, **your Sprint** |
| `robot_groundcontact.xml` | the parts that touch the floor when lying down | stand-up, sit, ground pick, roulade |
| `robot_groundcontact_rollers.xml` | plus passive wheels under the feet | roller tasks (Roller Sprint) |
| `robot_allcollisions.xml` | every part | not used by a task yet |
| `*_backlash.xml` | same, plus ±1° play per servo | `-Backlash-` tasks |

In the walk model only the feet can touch the ground, so a fallen duck sinks through the floor.
That's fine because training ends the episode the moment the duck tilts past 70°. It is also why
falling is "cheap" in this task, and why recovering from a fall is a different task
(`VelStand`).

Next: [2. Try it in simulation](02_TRY_IN_SIMULATION.md).
