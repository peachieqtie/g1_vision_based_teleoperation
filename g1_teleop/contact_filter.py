"""Candidate B (2026-09-15, measurement only): no hand <-> platform contact.

WHY
---
The synthetic teleop check found every permanent wrist jam is caused by a hand
catching the pickup platform; disabling platform collision removed all of them.
This is that control, made into a switchable configuration. The box still rests
on the platforms and still collides with the hands; only hand <-> platform is
removed. It is a MODELLING CHOICE, not a fix: the hand passes through the table,
which is physically false and must be disclosed.

HOW - three collision bits, no XML edit
---------------------------------------
Every colliding geom in this model is (contype 1, conaffinity 1) and there are no
explicit pairs or excludes (measured). MuJoCo collides a pair iff
`a.contype & b.conaffinity` or `b.contype & a.conaffinity` is nonzero, so:

    hands      (wrist links + pads)  contype 2   conaffinity 1
    platforms  (pickup + goal)       contype 4   conaffinity 1
    everything else                  contype 1   conaffinity 3

    hand     - platform   2&1=0, 4&1=0          EXCLUDED
    hand     - other      2&3=2                 collides (torso, box, floor, arm)
    platform - other      1&1=1                 collides (box, pelvis, legs)
    hand     - hand       2&1=0                 EXCLUDED  <- side effect, disclosed
    platform - platform   4&1=0                 excluded  (static, never touch)

Apply it BEFORE constructing `GraspWeld`: GraspWeld saves the pad bitmasks at
construction and restores them on release, so it must save the filtered ones.
"""
import mujoco

HAND_BODY_KEYS = ("wrist", "pad")
PLATFORM_GEOMS = ("platform_pickup_geom", "platform_goal_geom")


def apply_hand_platform_filter(model, platforms=PLATFORM_GEOMS):
    """Exclude hand<->platform contact in place. Returns what was changed.

    `platforms` defaults to BOTH. Measured 2026-09-15 that filtering the GOAL
    platform breaks the scripted place: the release jolts the box, and in the
    adopted configuration the hand coming to rest ON the goal platform is part
    of what keeps the box there. With that contact removed the hand falls
    through and the box is flung to the floor. Passing
    ("platform_pickup_geom",) scopes the filter to where the teleop hazard is
    and leaves the goal platform's support intact; a platform not listed is
    treated as an ordinary geom and still collides with the hands.
    """
    plats = set()
    for name in platforms:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise ValueError("platform geom not found: %s" % name)
        plats.add(g)
    hands, other = [], []
    for g in range(model.ngeom):
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        if (int(model.geom_contype[g]), int(model.geom_conaffinity[g])) != (1, 1):
            raise AssertionError(
                "geom %d has bitmask (%d,%d); this filter assumes every colliding "
                "geom starts at (1,1) and would silently mis-route anything else"
                % (g, model.geom_contype[g], model.geom_conaffinity[g]))
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[g])) or ""
        if g in plats:
            model.geom_contype[g], model.geom_conaffinity[g] = 4, 1
        elif any(k in body for k in HAND_BODY_KEYS):
            model.geom_contype[g], model.geom_conaffinity[g] = 2, 1
            hands.append(body)
        else:
            model.geom_contype[g], model.geom_conaffinity[g] = 1, 3
            other.append(body)
    return dict(hands=hands, platforms=len(plats), other=len(other))
