"""THROWAWAY DIAGNOSTIC - palm orientation at a plausible bimanual grasp pose.

Question this answers: at the pose the arms would actually hold while grasping
the box, which body axis of `{left,right}_wrist_yaw_link` points *inward* at
the box? That axis is the only candidate for the D7 slide-joint pad's slide
direction, and it must be knowable while the wrists stay pinned at
WRIST_NATURAL (D2).

Does NOT step physics, does NOT touch the ZED, does NOT modify any project
file. Loads scene.xml, drives the same 4-joint / 6-D elbow+wrist IK the live
teleop uses (D2/D3), calls mj_forward, prints frames. Delete when answered.

Usage:
  python tools/check_palm_orientation.py                     # full report + sweep
  python tools/check_palm_orientation.py --no-sweep --reach=0.28
  python tools/check_palm_orientation.py --no-sweep --wrist=-0.40,0.80,0.40

Note the =-form for negative values; argparse reads a bare "-0.40" as a flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g1_teleop import config as C          # noqa: E402
from g1_teleop.config import TeleopConfig  # noqa: E402
from g1_teleop.ik import solve_arm_ik      # noqa: E402
from g1_teleop.robot import G1Robot        # noqa: E402

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

BOX_Z = 0.84          # platform top 0.75 + box half-height 0.09 (BoxConfig.spawn_z)
BOX_HALF = 0.09       # box1_geom size


def elbow_from_two_link(shoulder, wrist, upper, fore, swivel_dir):
    """Place the elbow on the circle consistent with |S-E|=upper, |E-W|=fore.

    `swivel_dir` breaks the redundancy - it selects where on that circle the
    elbow sits (we want elbow low and outboard, matching IK_SEED_*). Returns
    (elbow_target, wrist_target_possibly_pulled_in).
    """
    v = wrist - shoulder
    d = np.linalg.norm(v)
    reach = upper + fore
    if d > reach * 0.995:                      # pull the wrist target inside reach
        wrist = shoulder + v * (reach * 0.995 / d)
        v = wrist - shoulder
        d = np.linalg.norm(v)
    u = v / d
    cos_a = np.clip((upper ** 2 + d ** 2 - fore ** 2) / (2 * upper * d), -1.0, 1.0)
    a = np.arccos(cos_a)
    perp = swivel_dir - np.dot(swivel_dir, u) * u
    n = np.linalg.norm(perp)
    perp = perp / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    return shoulder + upper * (np.cos(a) * u + np.sin(a) * perp), wrist


def deg(x):
    return np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))


def hand_geom_id(model, wrist_bid):
    """The rubber-hand geoms are unnamed in g1.xml, so find them by owning body.

    Returns the visual mesh geom attached to the wrist_yaw_link that is offset
    along local +x (the palm), or -1 if not found.
    """
    best, best_x = -1, 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != wrist_bid:
            continue
        if model.geom_pos[g][0] > best_x:
            best, best_x = g, float(model.geom_pos[g][0])
    return best


def hand_mesh_axes(model, data, hand_gid, wrist_bid):
    """Shape of the rubber-hand mesh, resolved into the wrist_yaw_link frame.

    CAREFUL: model.mesh_vert is NOT in the body frame. MuJoCo re-expresses mesh
    vertices in the mesh's own principal-axis frame and stores the compensating
    rotation in geom_quat (which is why g1.xml says quat="1 0 0 0" yet the
    compiled geom_quat is not identity). data.geom_xmat already folds that in,
    so world_vertex = geom_xpos + geom_xmat @ mesh_vert.

    A hand is THIN across the palm, LONG along the fingers. The thin axis is the
    palm-plane normal - the direction a palm pad must push along, and the axis
    the D7 slide joint needs. That is a property of the hand, independent of
    whichever axis happens to point at the box.

    Returns dict with extents and the thin/long axes expressed in BOTH the
    wrist_yaw_link body frame and world.
    """
    mid = int(model.geom_dataid[hand_gid])
    if mid < 0:
        return None
    adr = int(model.mesh_vertadr[mid])
    num = int(model.mesh_vertnum[mid])
    v = model.mesh_vert[adr:adr + num].reshape(-1, 3)
    size = v.max(axis=0) - v.min(axis=0)

    Rg = data.geom_xmat[hand_gid].reshape(3, 3)     # mesh frame -> world
    Rb = data.xmat[wrist_bid].reshape(3, 3)         # body frame -> world
    thin = int(np.argmin(size))
    long_ = int(np.argmax(size))

    def to_frames(i):
        ax = np.zeros(3)
        ax[i] = 1.0
        w = Rg @ ax
        return w, Rb.T @ w                          # world, body-frame

    n_w, n_b = to_frames(thin)
    f_w, f_b = to_frames(long_)
    j = int(np.argmax(np.abs(n_b)))
    return {
        "size": size, "thin": thin, "long": long_,
        "normal_world": n_w, "normal_body": n_b,
        "finger_world": f_w, "finger_body": f_b,
        "normal_body_axis": ("+" if n_b[j] >= 0 else "-") + "xyz"[j],
        "normal_body_purity": float(abs(n_b[j])),
        "palm_world": data.geom_xpos[hand_gid].copy(),
    }


def report_hand(side, model, data, wrist_bid, hand_gid, box_center):
    pos = data.xpos[wrist_bid].copy()
    R = data.xmat[wrist_bid].reshape(3, 3).copy()   # columns = body axes in world

    to_box = box_center - pos
    dist = np.linalg.norm(to_box)
    inward = to_box / dist

    dots = R.T @ inward                              # dot(inward, axis_j), j = x,y,z
    j = int(np.argmax(np.abs(dots)))
    sign = "+" if dots[j] >= 0 else "-"
    axis_name = "xyz"[j]
    angle = deg(abs(dots[j]))

    print("\n--- %s_wrist_yaw_link %s" % (side, "-" * 44))
    print("  world position (data.xpos) : %s" % pos)
    print("  rotation (data.xmat as 3x3; columns are the body axes in world):")
    for r in R:
        print("      %s" % r)
    print("    body x axis in world : %s" % R[:, 0])
    print("    body y axis in world : %s" % R[:, 1])
    print("    body z axis in world : %s" % R[:, 2])
    print("  box center                 : %s" % box_center)
    print("  wrist -> box distance      : %.4f m" % dist)
    print("  inward unit vector         : %s" % inward)
    print("  dot(inward, body x)        : %+.4f   -> %6.2f deg" % (dots[0], deg(dots[0])))
    print("  dot(inward, body y)        : %+.4f   -> %6.2f deg" % (dots[1], deg(dots[1])))
    print("  dot(inward, body z)        : %+.4f   -> %6.2f deg" % (dots[2], deg(dots[2])))
    print("  BEST-ALIGNED BODY AXIS     : %s%s   (|dot| = %.4f, %.2f deg off inward)"
          % (sign, axis_name, abs(dots[j]), angle))

    if hand_gid >= 0:
        hpos = data.geom_xpos[hand_gid].copy()
        h_in = box_center - hpos
        h_in = h_in / np.linalg.norm(h_in)
        h_dots = R.T @ h_in
        hj = int(np.argmax(np.abs(h_dots)))
        print("  --- same numbers taken from the PALM (rubber-hand geom), which is")
        print("      where a pad would actually sit, not the wrist link origin:")
        print("      palm world pos           : %s" % hpos)
        print("      offset from wrist (body) : %s" % (R.T @ (hpos - pos)))
        print("      palm -> box distance     : %.4f m" % np.linalg.norm(box_center - hpos))
        print("      inward unit vector       : %s" % h_in)
        print("      dot with body x / y / z  : %+.4f / %+.4f / %+.4f"
              % (h_dots[0], h_dots[1], h_dots[2]))
        print("      BEST-ALIGNED BODY AXIS   : %s%s  (|dot| = %.4f, %.2f deg off inward)"
              % ("+" if h_dots[hj] >= 0 else "-", "xyz"[hj],
                 abs(h_dots[hj]), deg(abs(h_dots[hj]))))
    return inward, R, sign + axis_name, angle, dots


def pose_arms(robot, cfg, wrist_angles, box_center, sep, target="palm", iters=4):
    """Reset, pin the wrists at `wrist_angles`, then run the real 4-joint IK.

    The wrist roll/pitch/yaw joints sit ABOVE wrist_yaw_link in the kinematic
    chain, so changing them moves and rotates the very body the IK is aiming at
    - which is exactly why re-tuning WRIST_NATURAL is a live option here.

    target="wrist": put wrist_yaw_link's origin on the box face.
    target="palm":  put the RUBBER HAND on the box face. The palm is ~10.6 cm
      distal of the wrist origin, so these are very different poses and only the
      second is a real grasp. The IK can only aim at a body origin, so we solve,
      measure the palm's world offset from the wrist, shift the wrist target by
      minus that offset, and re-solve until it settles.
    """
    model, data = robot.model, robot.data
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for name, angle in wrist_angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = angle
    data.qpos[robot.box_qadr:robot.box_qadr + 3] = box_center
    data.qpos[robot.box_qadr + 3:robot.box_qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    half = sep / 2.0
    out = {}
    for side, sign in (("left", +1.0), ("right", -1.0)):
        sh = robot.left_shoulder_world() if side == "left" else robot.right_shoulder_world()
        upper = robot.upper_arm_left if side == "left" else robot.upper_arm_right
        fore = robot.forearm_left if side == "left" else robot.forearm_right
        el_b = robot.left_elbow_body if side == "left" else robot.right_elbow_body
        wr_b = robot.left_wrist_body if side == "left" else robot.right_wrist_body
        qp = robot.ik_left_qpos if side == "left" else robot.ik_right_qpos
        dof = robot.ik_left_dof if side == "left" else robot.ik_right_dof
        lim = robot.ik_left_lim if side == "left" else robot.ik_right_lim
        neu = robot.neutral_left if side == "left" else robot.neutral_right

        hand_g = hand_geom_id(model, wr_b)
        goal = box_center + np.array([0.0, sign * half, 0.0])
        swivel = np.array([0.0, sign * 0.35, -1.0])

        wr_goal = goal.copy()
        for _ in range(iters if target == "palm" else 1):
            el_t, wr_t = elbow_from_two_link(sh, wr_goal, upper, fore, swivel)
            q = solve_arm_ik(model, data, el_b, wr_b, el_t, wr_t,
                             qp, dof, lim, neu, cfg.ik)
            if target != "palm" or hand_g < 0:
                break
            palm_offset = data.geom_xpos[hand_g] - data.xpos[wr_b]
            wr_goal = goal - palm_offset

        palm_err = (float(np.linalg.norm(data.geom_xpos[hand_g] - goal))
                    if hand_g >= 0 else float("nan"))
        out[side] = {
            "q": q, "el_target": el_t, "wr_target": wr_t, "goal": goal,
            "wr_res": float(np.linalg.norm(data.xpos[wr_b] - wr_t)),
            "el_res": float(np.linalg.norm(data.xpos[el_b] - el_t)),
            "palm_err": palm_err,
        }
    mujoco.mj_forward(model, data)
    return out


def palm_alignment(model, data, wrist_bid, hand_gid, box_center):
    """How squarely does the FLAT OF THE HAND face the box?

    Returns (dot, body-frame axis label of the pressing normal, palm->box unit).
    dot = +1 means the palm plane is normal to the line to the box, i.e. a pad
    on that face would press flush. This is the metric that decides D7, not
    "some body axis happens to point at the box".
    """
    h = hand_mesh_axes(model, data, hand_gid, wrist_bid)
    if h is None:
        return 0.0, "?", np.zeros(3)
    v = box_center - h["palm_world"]
    v = v / np.linalg.norm(v)
    s = 1.0 if float(h["normal_world"] @ v) >= 0 else -1.0
    n_b = s * h["normal_body"]
    j = int(np.argmax(np.abs(n_b)))
    return float(s * h["normal_world"] @ v), \
        ("+" if n_b[j] >= 0 else "-") + "xyz"[j], v


def sweep_wrists(robot, cfg, box_center, sep, target="palm"):
    """Is the misalignment a property of the arm, or just of the chosen
    WRIST_NATURAL values? Sweep roll and yaw (mirrored L/R, pitch held) and see
    how well any palm axis can be aimed at the box."""
    model, data = robot.model, robot.data
    lh = hand_geom_id(model, robot.left_wrist_body)
    rh = hand_geom_id(model, robot.right_wrist_body)


    grid = np.arange(-1.6, 1.61, 0.4)

    def score(roll, pitch_, yaw):
        wa = {
            "left_wrist_roll_joint": float(roll),
            "left_wrist_pitch_joint": float(pitch_),
            "left_wrist_yaw_joint": float(yaw),
            "right_wrist_roll_joint": float(-roll),
            "right_wrist_pitch_joint": float(pitch_),
            "right_wrist_yaw_joint": float(-yaw),
        }
        res = pose_arms(robot, cfg, wa, box_center, sep, target=target)
        dl, axl, _ = palm_alignment(model, data, robot.left_wrist_body, lh, box_center)
        dr, axr, _ = palm_alignment(model, data, robot.right_wrist_body, rh, box_center)
        return (min(dl, dr), roll, pitch_, yaw, dl, axl, dr, axr,
                max(res["left"]["palm_err"], res["right"]["palm_err"]))

    rows = [score(r, p, y) for r in grid for p in grid for y in grid]
    rows.sort(key=lambda r: -r[0])
    cur = score(C.WRIST_NATURAL["left_wrist_roll_joint"],
                C.WRIST_NATURAL["left_wrist_pitch_joint"],
                C.WRIST_NATURAL["left_wrist_yaw_joint"])

    print("\n" + "=" * 70)
    print("WRIST_NATURAL SWEEP - can the palms be made to face each other?")
    print("=" * 70)
    print("  roll/pitch/yaw swept over [-1.6, 1.6] rad (joint ranges +-1.61 / +-1.97),")
    print("  mirrored L/R. Score = worse of the two hands' dot(palm normal, palm->box);")
    print("  1.0 = the flat of both hands presses the box squarely.")
    print("  %7s %7s %7s %7s %7s %7s %7s %7s %11s"
          % ("score", "roll", "pitch", "yaw", "dotL", "axisL", "dotR", "axisR", "palm_err_mm"))
    for r in rows[:10]:
        print("  %7.4f %7.2f %7.2f %7.2f %7.4f %7s %7.4f %7s %11.1f"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8] * 1000.0))
    b = rows[0]
    print("\n  BEST    : roll=%+.2f pitch=%+.2f yaw=%+.2f -> %.4f (%.2f deg off flush), "
          "press axes L=%s R=%s" % (b[1], b[2], b[3], b[0], deg(b[0]), b[5], b[7]))
    print("  CURRENT : roll=%+.2f pitch=%+.2f yaw=%+.2f -> %.4f (%.2f deg off flush), "
          "press axes L=%s R=%s" % (cur[1], cur[2], cur[3], cur[0], deg(cur[0]),
                                    cur[5], cur[7]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", type=float, default=0.32,
                    help="forward distance from pelvis x to the box center (m)")
    ap.add_argument("--sep", type=float, default=0.18,
                    help="left-right separation between the two wrist targets (m)")
    ap.add_argument("--target", choices=("palm", "wrist"), default="palm",
                    help="place the palm (real grasp) or the wrist link origin "
                         "on the box faces")
    ap.add_argument("--wrist", default=None, metavar="ROLL,PITCH,YAW",
                    help="override WRIST_NATURAL for the LEFT hand (right is "
                         "mirrored), to re-check the report at a candidate pose")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the WRIST_NATURAL roll/pitch/yaw sweep")
    args = ap.parse_args()

    cfg = TeleopConfig()
    robot = G1Robot(cfg)            # loads scene.xml, applies WRIST_NATURAL, mj_forward
    model, data = robot.model, robot.data

    # ---- 1. choose a plausible grasp geometry -------------------------------
    # The box lives at x=1.5 in the scene; the robot only reaches it after
    # walking there. For a kinematics-only check, put the grasp target where the
    # arms can actually reach, and move box1 there so the printed vectors mean
    # something.
    pelvis = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].copy()
    ls = robot.left_shoulder_world()
    rs = robot.right_shoulder_world()
    box_center = np.array([pelvis[0] + args.reach, 0.5 * (ls[1] + rs[1]), BOX_Z])

    data.qpos[robot.box_qadr:robot.box_qadr + 3] = box_center
    data.qpos[robot.box_qadr + 3:robot.box_qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    half = args.sep / 2.0

    print("=" * 70)
    print("PALM ORIENTATION DIAGNOSTIC (kinematic; mj_forward only, no stepping)")
    print("=" * 70)
    print("  model                : %s" % cfg.model_path)
    print("  pelvis world pos     : %s" % pelvis)
    print("  L / R shoulder world : %s / %s" % (ls, rs))
    print("  upper/fore (L)       : %.4f / %.4f m" % (robot.upper_arm_left, robot.forearm_left))
    print("  upper/fore (R)       : %.4f / %.4f m" % (robot.upper_arm_right, robot.forearm_right))
    print("  box center (grasp)   : %s   (half-width %.2f m)" % (box_center, BOX_HALF))
    print("  grasp goals    L / R : %s / %s"
          % (box_center + np.array([0.0, +half, 0.0]),
             box_center + np.array([0.0, -half, 0.0])))
    print("  IK aims the %s at those goals" % args.target.upper())
    print("  wrists pinned (D2)   : %s" % dict(C.WRIST_NATURAL))

    # ---- 2. drive the same IK the live teleop uses --------------------------
    # Swivel preference matches IK_SEED_*: elbow low and outboard.
    wrist_angles = dict(C.WRIST_NATURAL)
    if args.wrist:
        r_, p_, y_ = (float(t) for t in args.wrist.split(","))
        wrist_angles = {
            "left_wrist_roll_joint": r_, "left_wrist_pitch_joint": p_,
            "left_wrist_yaw_joint": y_, "right_wrist_roll_joint": -r_,
            "right_wrist_pitch_joint": p_, "right_wrist_yaw_joint": -y_,
        }
        print("  OVERRIDE wrists      : %s" % wrist_angles)
    res = pose_arms(robot, cfg, wrist_angles, box_center, args.sep,
                    target=args.target)
    for side in ("left", "right"):
        r = res[side]
        print("\n  [%s] elbow target %s  wrist target %s"
              % (side, r["el_target"], r["wr_target"]))
        print("  [%s] IK solution (sh_pitch, sh_roll, sh_yaw, elbow) = %s" % (side, r["q"]))
        print("  [%s] wrist residual = %.1f mm   elbow residual = %.1f mm   "
              "palm-to-goal = %.1f mm"
              % (side, r["wr_res"] * 1000.0, r["el_res"] * 1000.0,
                 r["palm_err"] * 1000.0))

    # ---- 3. report the two palm frames --------------------------------------
    lh = hand_geom_id(model, robot.left_wrist_body)
    rh = hand_geom_id(model, robot.right_wrist_body)

    in_l, R_l, ax_l, ang_l, dots_l = report_hand(
        "left", model, data, robot.left_wrist_body, lh, box_center)
    in_r, R_r, ax_r, ang_r, dots_r = report_hand(
        "right", model, data, robot.right_wrist_body, rh, box_center)

    # ---- 4. verdict ---------------------------------------------------------
    facing = float(np.dot(in_l, in_r))
    sep = float(np.linalg.norm(data.xpos[robot.left_wrist_body]
                               - data.xpos[robot.right_wrist_body]))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("  actual wrist-to-wrist separation : %.4f m (box width %.2f m)"
          % (sep, 2 * BOX_HALF))
    print("  dot(inward_L, inward_R)          : %+.4f  (%.2f deg apart; -1.0 = "
          "perfectly opposed, i.e. palms facing each other)" % (facing, deg(facing)))
    print("  left  inward axis  : %s  (%.2f deg off)" % (ax_l, ang_l))
    print("  right inward axis  : %s  (%.2f deg off)" % (ax_r, ang_r))
    same = ax_l == ax_r
    mirrored = (ax_l[1] == ax_r[1]) and (ax_l[0] != ax_r[0])
    print("  same axis AND sign on both hands? %s      mirrored (same letter, "
          "opposite sign)? %s" % (same, mirrored))
    print("\n  Alignment table (dot of the inward direction with each body axis):")
    print("    %-6s %9s %9s %9s" % ("hand", "x", "y", "z"))
    print("    %-6s %+9.4f %+9.4f %+9.4f" % ("left", dots_l[0], dots_l[1], dots_l[2]))
    print("    %-6s %+9.4f %+9.4f %+9.4f" % ("right", dots_r[0], dots_r[1], dots_r[2]))
    # ---- 4b. where does the PALM SURFACE actually face? ---------------------
    print("\n" + "=" * 70)
    print("HAND MESH SHAPE - which local axis is the palm normal?")
    print("=" * 70)
    normals = {}
    for side, gid, bid in (("left", lh, robot.left_wrist_body),
                           ("right", rh, robot.right_wrist_body)):
        h = hand_mesh_axes(model, data, gid, bid) if gid >= 0 else None
        if h is None:
            print("  %s: no mesh data" % side)
            continue
        # Orient the normal so it points toward the box (the pressing direction).
        to_box = box_center - h["palm_world"]
        to_box = to_box / np.linalg.norm(to_box)
        s = 1.0 if float(h["normal_world"] @ to_box) >= 0 else -1.0
        n_w, n_b = s * h["normal_world"], s * h["normal_body"]
        j = int(np.argmax(np.abs(n_b)))
        d = float(n_w @ to_box)
        normals[side] = n_w

        print("  %s hand mesh (principal-frame extents %s m):" % (side, h["size"]))
        print("    thinnest %.4f m -> PALM-PLANE NORMAL, body frame %s"
              % (h["size"][h["thin"]], h["normal_body"]))
        print("    longest  %.4f m -> FINGER direction,   body frame %s"
              % (h["size"][h["long"]], h["finger_body"]))
        print("    pressing normal (signed toward box), body frame : %s" % n_b)
        print("    nearest body axis  : %s%s  (purity %.4f)"
              % ("+" if n_b[j] >= 0 else "-", "xyz"[j], abs(n_b[j])))
        print("    same, in world     : %s" % n_w)
        print("    dot(palm normal, palm->box) : %+.4f  (%.2f deg off)" % (d, deg(d)))
    print("  (+1 = the flat of the hand squarely faces the box)")
    if len(normals) == 2:
        opp = float(normals["left"] @ normals["right"])
        print("  dot(left press normal, right press normal) : %+.4f  (%.2f deg apart)"
              % (opp, deg(opp)))
        print("  -1.0 here = the two flat faces are exactly opposed, i.e. the palms")
        print("  face each other and a pad on each would squeeze the box.")

    print("\n  The pad slide joint must use the PRESSING NORMAL above (section 4b),")
    print("  not whichever wrist-link axis happens to point at the box (section 4).")
    print("  Those differ, and only the first presses with the flat of the hand.")

    if not args.no_sweep:
        sweep_wrists(robot, cfg, box_center, args.sep, target=args.target)


if __name__ == "__main__":
    main()
