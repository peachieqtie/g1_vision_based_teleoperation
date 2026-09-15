"""Episode recorder: one .npz per demonstration, built through `spec.SpecLayout`.

OBSERVATION ONLY
----------------
The recorder never writes to the model, never changes a control, and never
touches `scripted_demo.py`. It attaches by wrapping `mujoco.mj_step` and reading
the demonstrator's own frame locals - the way the 2026-09-10 constant-dimension
audit was instrumented - because the gates are only meaningful if recording a
run and not recording it produce the same run. `tools/record_episodes.py
--invariance` measures exactly that.

TIMING (spec.py, TIMING CONVENTION)
-----------------------------------
A tick is taken when the demonstrator's loop index is a multiple of
`spec.PHYSICS_STEPS_PER_TICK` (20 steps = 25 Hz, D4), at `mj_step` ENTRY. That
is the one instant where `data.ctrl` holds the command about to be applied and
the state is the one it is applied from, so `state[t]` and `action[t]` are the
pair "what the demonstrator saw, what it did about it". Both come from
`SpecLayout.build`; nothing here hand-assembles a vector.

WHAT IS LOGGED THAT THE 47-D STATE DOES NOT CARRY, AND WHY
----------------------------------------------------------
RAW `qpos` and `qvel` per tick. They are INSURANCE, not a policy input, and they
are the reason the state design is the one dataset decision that cannot be
regretted: almost anything one later wishes were in the 47-D vector is derivable
from raw state plus an observation window (velocity is a finite difference), but
LEG JOINT ANGLES are not - proposal 3.4 excludes them, and once 150 episodes are
collected without them they are gone.

`gait_phase` is logged as metadata and is NEVER a policy input: it is a clock,
and handing every policy a clock gives BC the temporal capability ACT-LSTM is
meant to supply (CLAUDE.md section 8, schema freeze).

`contact_contract` records whether the D18 hand/pickup-platform exclusion was
active, read from the compiled model rather than from a flag. If collection and
evaluation disagree on it, every policy is evaluated in physics it never
learned, and nothing else in the file would reveal that.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import mujoco

from g1_data import spec
from g1_data.reset import state_fingerprint
from g1_teleop.contact_contract import contract_of
from g1_teleop.indices import ModelIndex

TICK = spec.PHYSICS_STEPS_PER_TICK
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=ROOT, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                       # noqa: BLE001
        return "unknown"


class EpisodeBuffer:
    """Per-tick rows held in memory; written once, at the end.

    A rejected episode must cost nothing, so nothing reaches the disk until
    `save` is called - which the operator only reaches after ACCEPT.
    """

    def __init__(self, model, ix: Optional[ModelIndex] = None,
                 sp: Optional[spec.SpecLayout] = None):
        self.ix = ix or ModelIndex.resolve(model)
        self.sp = sp or spec.SpecLayout.resolve(model, self.ix)
        self.states, self.actions, self.phases = [], [], []
        self.gait, self.qpos, self.qvel, self.ticks = [], [], [], []
        self.t_wall0 = time.perf_counter()

    def tick(self, model, data, act, cmd, phase_label: int, gait_counter: int,
             step_index: int, dt: float, gait_period: float) -> None:
        """One recorded tick. Call at `mj_step` entry, never after it."""
        s, a = self.sp.build(model, data, self.ix, act=act, cmd=cmd, sync=True)
        self.states.append(s)
        self.actions.append(a)
        self.phases.append(int(phase_label))
        phz = ((gait_counter * dt) % gait_period) / gait_period
        self.gait.append((np.sin(2 * np.pi * phz), np.cos(2 * np.pi * phz)))
        self.qpos.append(np.array(data.qpos, dtype=np.float64))
        self.qvel.append(np.array(data.qvel, dtype=np.float64))
        self.ticks.append(int(step_index))

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str, meta: dict) -> str:
        """Write the episode. Float32 arrays, JSON metadata."""
        if not self.states:
            raise ValueError("refusing to write an episode with no ticks")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        m = dict(meta)
        m.setdefault("spec_version", spec.SPEC_VERSION)
        m.setdefault("git_commit", _git_commit())
        m.setdefault("mujoco_version", mujoco.__version__)
        m.setdefault("python_version", sys.version.split()[0])
        m.setdefault("n_ticks", len(self.states))
        np.savez_compressed(
            path,
            states=np.asarray(self.states, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            phase_labels=np.asarray(self.phases, dtype=np.int8),
            gait_phase=np.asarray(self.gait, dtype=np.float32),
            qpos=np.asarray(self.qpos, dtype=np.float32),
            qvel=np.asarray(self.qvel, dtype=np.float32),
            step_index=np.asarray(self.ticks, dtype=np.int64),
            meta=np.array(json.dumps(m)))
        return path


def load_episode(path: str):
    """(arrays, meta). Refuses a file written under another spec version."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        spec.assert_spec_version(meta["spec_version"], where=os.path.basename(path))
        arrays = {k: z[k].copy() for k in z.files if k != "meta"}
    return arrays, meta


class ScriptedRecorder:
    """Records `scripted_demo.run_episode` without modifying it.

    A context manager around the call. Two things are wrapped, both read-only:

      `mujoco.mj_step`              the tick hook. It fires only for frames
                                    belonging to `run_episode`, so a step taken
                                    anywhere else could never be mistaken for a
                                    demonstrator tick.
      `scripted_demo.reset_episode` captures the post-reset fingerprint, which is
                                    what the replay test asserts against BEFORE it
                                    blames the actions for a divergence.

    `act` is the velocity command in force for the step about to be taken. The
    demonstrator computes it AFTER `mj_step`, for the next step, so at tick 0 it
    does not exist yet and zeros are logged - which is the truth: no locomotion
    command had been issued. While the base lock is engaged `build_action` zeroes
    those dims anyway (D17), reading the lock from the model's own equality
    rather than from anything passed in here.
    """

    def __init__(self):
        self.buf: Optional[EpisodeBuffer] = None
        self.reset_fingerprint = ""
        self.weld_engage_tick = -1
        self.weld_release_tick = -1
        self.lock_engage_tick = -1
        self.lock_release_tick = -1
        self.box_pre_grasp_disturb_mm = 0.0
        self._box0 = None
        self._prev_weld = False
        self._prev_lock = False

    # ---- attachment ----------------------------------------------------
    def __enter__(self) -> "ScriptedRecorder":
        from g1_data import scripted_demo as SD
        self._sd = SD
        self._real_step = mujoco.mj_step
        self._real_reset = SD.reset_episode
        rec = self

        def reset_wrapper(model, data, index, cfg, seed, *a, **k):
            start = rec._real_reset(model, data, index, cfg, seed, *a, **k)
            rec.reset_fingerprint = state_fingerprint(data, index)
            return start

        def step_wrapper(m, d, *a, **k):
            f = sys._getframe(1)
            if f.f_code.co_name == "run_episode":
                # A recorder fault must not corrupt a collection session
                # silently: let it raise, and fail the episode loudly.
                rec._observe(m, d, f.f_locals)
            return rec._real_step(m, d, *a, **k)

        mujoco.mj_step = step_wrapper
        SD.reset_episode = reset_wrapper
        return self

    def __exit__(self, *exc) -> None:
        mujoco.mj_step = self._real_step
        self._sd.reset_episode = self._real_reset

    # ---- the tick ------------------------------------------------------
    def _observe(self, m, d, L: dict) -> None:
        i, ix, weld = L["i"], L["ix"], L["weld"]
        if self.buf is None:
            self.buf = EpisodeBuffer(m, ix)
            # The model the episode actually ran on: the contact contract in the
            # metadata is read back from THIS model, never from a config flag.
            self.model = m
            self._weld_eq = weld.eq_id
            self._lock_eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                              "base_lock")
            self._box0 = np.array(d.qpos[ix.box_qpos][:3], dtype=np.float64)
        welded = bool(d.eq_active[self._weld_eq])
        locked = bool(d.eq_active[self._lock_eq])
        tick = i // TICK
        if welded and not self._prev_weld and self.weld_engage_tick < 0:
            self.weld_engage_tick = tick
        if self._prev_weld and not welded and self.weld_release_tick < 0:
            self.weld_release_tick = tick
        if locked and not self._prev_lock and self.lock_engage_tick < 0:
            self.lock_engage_tick = tick
        if self._prev_lock and not locked and self.lock_release_tick < 0:
            self.lock_release_tick = tick
        self._prev_weld, self._prev_lock = welded, locked
        if self.weld_engage_tick < 0:
            # How far the box moved BEFORE THE FIRST GRASP. Bounded by the
            # first engage, not by "not welded": after RELEASE at the goal the
            # box is legitimately 1.4 m from its spawn, and counting that read
            # 1411 mm on a healthy episode. A demonstration that shoves the box
            # before grasping it is a demonstration of a different task, and O26
            # plus the D18 teleop finding both make that a live risk.
            moved = np.linalg.norm(np.asarray(d.qpos[ix.box_qpos][:3]) - self._box0)
            self.box_pre_grasp_disturb_mm = max(self.box_pre_grasp_disturb_mm,
                                                1000.0 * float(moved))
        if i % TICK:
            return
        act = L.get("act")
        act = np.zeros(3) if act is None else np.asarray(act, dtype=np.float64)
        loco = L["cfg"].loco
        self.buf.tick(m, d, act=act, cmd=float(L.get("cmd", 0.0)),
                      phase_label=int(L["phase"]),
                      gait_counter=int(L["carry"].counter),
                      step_index=i, dt=loco.sim_dt, gait_period=loco.gait_period)

    # ---- the file ------------------------------------------------------
    def metadata(self, model, result: dict, cfg, demo, seed: int,
                 extra: Optional[dict] = None) -> dict:
        """Episode metadata. Every value measured or read back, none asserted."""
        wall = time.perf_counter() - self.buf.t_wall0
        sim_s = float(result["n_steps"]) * cfg.loco.sim_dt
        box0 = np.asarray(result["box_start"], dtype=float)
        m = dict(
            seed=int(seed),
            box_spawn_xy=[float(box0[0]), float(box0[1])],
            heldout=bool(result["heldout"]),
            standoff_cmd=float(demo.standoff),
            lateral_cmd=float(result.get("lock_lateral", float("nan"))),
            # Outcome as the DEMONSTRATOR reports it. The four canonical
            # per-phase success flags are the next chunk (they need the phase
            # labeller and the success module); everything they are computed
            # from is already in this file.
            outcome=dict(
                ok=bool(result["ok"]), fail_phase=result["fail_phase"],
                engaged=bool(result["engaged"]), resting=bool(result["resting"]),
                fell=bool(result["fell"]),
                placement_error=float(result["place_err_m"]),
                tilt_deg=float(result["tilt_deg"]),
                palm_err_mm=float(result["palm_err_mm"]),
                carry_drift_mm=float(result["carry_drift_mm"]),
                max_pitch_deg=float(result["max_pitch_deg"])),
            weld_engage_tick=self.weld_engage_tick,
            weld_release_tick=self.weld_release_tick,
            lock_engage_tick=self.lock_engage_tick,
            lock_release_tick=self.lock_release_tick,
            wrist_dev_rad=float(result["wrist_dev_rad"]),
            wrist_dev_joint=result["wrist_dev_joint"],
            # TR19: a POSITIVE clearance cannot be measured in this build, so
            # this is penetration DEPTH - negative while the hand is inside the
            # slab, 0.0 when it never touched. It is not "clearance = 0".
            min_hand_platform_clearance_mm=float(result["hand_plat_depth_mm"]),
            hand_platform_contacts=int(result["hand_plat_contacts"]),
            min_box_platform_clearance_m=float(result["min_clearance_m"]),
            box_pre_grasp_disturb_mm=float(self.box_pre_grasp_disturb_mm),
            sim_to_wall_ratio=float(sim_s / wall) if wall > 0 else float("nan"),
            sim_seconds=sim_s, wall_seconds=float(wall),
            n_steps=int(result["n_steps"]),
            reset_fingerprint=self.reset_fingerprint,
            contact_contract=contract_of(model),
            lock_predicate=bool(demo.lock_predicate),
            source="scripted_demo.run_episode",
            n_ticks=len(self.buf),
        )
        if extra:
            m.update(extra)
        return m
