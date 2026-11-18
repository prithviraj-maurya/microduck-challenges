"""Shared helpers for the learning examples: load the duck, drive its servos.

The robot is the same MJCF that training uses (`scene_walk.xml` from the installed
mjlab_microduck package). Its 14 servos are converted to torque motors and driven
by the Arena's numpy copy of the BAM XL330 model (`arena.actuator`), which is the
same servo physics training and the Arena use, at nominal values.

Run the examples with the workshop group, which installs the `arena` package:

    uv run --group workshop python docs/learn/examples/<example>.py --help
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
from arena.actuator import ARMATURE, FORCE_LIMIT, SOLIMP_FRICTION, SOLREF_FRICTION, Servos
from arena.constants import DEFAULT_POSE, JOINT_NAMES, TIMESTEP

# Find the package folder without importing it (importing pulls in torch and warp).
_PKG = Path(importlib.util.find_spec("mjlab_microduck").submodule_search_locations[0])
SCENE = _PKG / "robot" / "microduck" / "scene_walk.xml"


def load_duck() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The walk scene with the 14 servos as BAM torque motors, as the Arena builds it."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-FORCE_LIMIT, FORCE_LIMIT)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
    for joint in spec.joints:
        if joint.name in JOINT_NAMES:
            joint.armature = ARMATURE
            joint.damping = np.zeros((3, 1))
            joint.frictionloss = 0.0
            joint.solref_friction = SOLREF_FRICTION
            joint.solimp_friction = SOLIMP_FRICTION
    model = spec.compile()
    model.opt.timestep = TIMESTEP
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    return model, data


def servos_for(model: mujoco.MjModel, seed: int = 0) -> tuple[Servos, np.ndarray, np.ndarray]:
    """A BAM servo driver for the 14 joints, plus their qpos and dof addresses."""
    jids = np.array([model.joint(n).id for n in JOINT_NAMES])
    dof = model.jnt_dofadr[jids]
    qpos = model.jnt_qposadr[jids]
    act_ids = np.array([model.actuator(n).id for n in JOINT_NAMES])
    return Servos(model, dof, act_ids, np.random.default_rng(seed)), qpos, dof


def trunk_tilt_deg(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Angle between the trunk's up axis and the world's up axis."""
    up_z = data.xmat[model.body("trunk_base").id].reshape(3, 3)[2, 2]
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def control_step(model, data, servos: Servos, target: np.ndarray, substeps: int = 4) -> None:
    """One 50 Hz control step: hold `target` for 4 physics substeps of 5 ms."""
    for _ in range(substeps):
        servos.drive(data, target)
        mujoco.mj_step(model, data)


__all__ = ["DEFAULT_POSE", "JOINT_NAMES", "load_duck", "servos_for", "trunk_tilt_deg", "control_step"]
