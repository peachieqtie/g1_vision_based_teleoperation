"""Chapter 3 figure: the pickup platform, the box sample region, and the
Objective 4 held-out patch, in world XY.

    python tools/plot_platform_regions.py

Writes `docs/figures/platform_regions.{pdf,png}`.

EVERYTHING IS READ FROM THE MODEL OR FROM CONFIG - there is not one hand-placed
coordinate in this file. The platform footprint comes from
`box_reset.platform_extent`, which resolves `platform_pickup_geom` by name, and
the box footprint from `box_reset.box_half_extent`. That is deliberate: a Q6
platform or box resize must move the figure rather than silently invalidate it,
which is the same reason those helpers exist at all (O3).

The scatter is not decoration either. It is 2000 real draws through
`box_reset.sample_box_pose`, coloured by `box_reset.in_heldout` - the same two
functions the recorder and the end-of-collection leak check use. If the sampler
ever stops agreeing with the geometry, this figure is what shows it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_teleop import box_reset as BR
from g1_teleop.config import BoxConfig, GraspConfig, MODEL_PATH

N_SEEDS = 2000
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "figures")

# A restrained palette that survives greyscale printing: the two scatter
# populations differ in lightness as well as hue.
C_PLATFORM = "#2f2f2f"
C_REGION = "#1f6fb2"
C_HELDOUT = "#c0392b"
C_TRAIN_PT = "#7fb3d5"
C_HELD_PT = "#c0392b"
C_REACH = "#7d3c98"


def main() -> int:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    box = BoxConfig()
    grasp = GraspConfig()

    # Guards first: if the region does not fit the platform, or the held-out
    # patch is not inside the region, the figure would be drawing a lie.
    BR.assert_spawn_fits(model, box)
    BR.assert_heldout_inside_region(box)

    plat_c, plat_h = BR.platform_extent(model, box)
    box_h = BR.box_half_extent(model, box)
    reg_c = np.asarray(box.pickup_center, dtype=float)
    reg_h = np.asarray(box.pickup_half, dtype=float)
    (hx0, hx1), (hy0, hy1) = box.heldout_x, box.heldout_y
    reach_x = grasp.max_base_x + grasp.grasp_max

    # ---- the sample, through the real sampler --------------------------------
    pts = np.array([BR.sample_box_pose(box, s)[0][:2] for s in range(N_SEEDS)])
    held = np.array([BR.in_heldout(p, box) for p in pts])
    frac = float(held.mean())
    predicted = ((hx1 - hx0) * (hy1 - hy0)) / (4.0 * reg_h[0] * reg_h[1])

    print("platform  centre %s  half %s   (from %s)"
          % (np.round(plat_c, 4), np.round(plat_h, 4), box.platform_geom))
    print("box half-extent %s   (from %s)" % (np.round(box_h, 4), box.box_geom))
    print("sample region  x [%.3f, %.3f]  y [%.3f, %.3f]"
          % (reg_c[0] - reg_h[0], reg_c[0] + reg_h[0],
             reg_c[1] - reg_h[1], reg_c[1] + reg_h[1]))
    print("held-out patch x [%.3f, %.3f]  y [%.3f, %.3f]" % (hx0, hx1, hy0, hy1))
    print("reach limit    x  %.3f  (max_base_x %.3f + grasp_max %.3f)"
          % (reach_x, grasp.max_base_x, grasp.grasp_max))
    print("held-out fraction: measured %.4f (%d/%d) vs geometry %.4f"
          % (frac, int(held.sum()), N_SEEDS, predicted))
    err = abs(frac - predicted)
    # 2000 Bernoulli draws at p=0.143 have sd = 0.0078, so 3 sd is 0.024.
    sd = float(np.sqrt(predicted * (1 - predicted) / N_SEEDS))
    print("  difference %.4f = %.2f sd of a %d-draw binomial (sd %.4f)"
          % (err, err / sd, N_SEEDS, sd))
    if err > 4 * sd:
        print("  *** DISAGREEMENT: the figure is drawn from the geometry and "
              "the scatter from the sampler. They do not match, so the SAMPLER "
              "or the config is wrong, not this figure. ***")

    # ---- figure --------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed real fonts, not paths
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    ax.add_patch(Rectangle(plat_c - plat_h, 2 * plat_h[0], 2 * plat_h[1],
                           facecolor="#f2f2f2", edgecolor=C_PLATFORM,
                           linewidth=1.4, zorder=1,
                           label="pickup platform (from model)"))
    ax.add_patch(Rectangle(reg_c - reg_h, 2 * reg_h[0], 2 * reg_h[1],
                           facecolor="none", edgecolor=C_REGION, linewidth=1.6,
                           linestyle="--", zorder=4, label="box sample region"))
    ax.add_patch(Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0,
                           facecolor=C_HELDOUT, alpha=0.14, edgecolor=C_HELDOUT,
                           linewidth=1.6, zorder=3,
                           label="held-out patch (Objective 4)"))

    ax.scatter(pts[~held, 0], pts[~held, 1], s=2.0, c=C_TRAIN_PT, alpha=0.55,
               linewidths=0, zorder=2,
               label="training spawns (%d)" % int((~held).sum()))
    ax.scatter(pts[held, 0], pts[held, 1], s=2.6, c=C_HELD_PT, alpha=0.85,
               linewidths=0, zorder=5,
               label="held-out spawns (%d, %.1f%%)" % (int(held.sum()),
                                                       100 * frac))

    # box footprint at the far corner of the region: this is what edge_margin
    # buys, drawn rather than asserted
    corner = reg_c + reg_h
    ax.add_patch(Rectangle(corner - box_h, 2 * box_h[0], 2 * box_h[1],
                           facecolor="none", edgecolor="#000000", linewidth=1.1,
                           zorder=6, label="box footprint at a corner spawn"))
    # `edge_margin` is a MINIMUM, and only y is actually sitting on it: x has
    # 0.19 - 0.06 - 0.09 = 0.04 m to spare. Annotate the binding axis, and take
    # the number from the measured gap rather than from the config, so a config
    # change that does not reach the geometry cannot go unnoticed here.
    gap = (plat_c + plat_h) - (corner + box_h)
    ann_x = corner[0] + box_h[0]
    ax.annotate("", xy=(ann_x, plat_c[1] + plat_h[1]),
                xytext=(ann_x, corner[1] + box_h[1]),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8),
                zorder=7)
    # The gap is 20 mm - about 2% of the y axis - so a label sitting on it would
    # collide with the platform edge. Park it in the empty platform area to the
    # left and lead the eye across.
    ax.annotate("edge margin %.0f mm" % (1000 * gap[1]),
                xy=(ann_x, corner[1] + box_h[1] + gap[1] / 2), xycoords="data",
                xytext=(plat_c[0] - plat_h[0] + 0.015,
                        plat_c[1] + plat_h[1] - 0.055), textcoords="data",
                fontsize=7, color="#555555", va="center", ha="left", zorder=7,
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7,
                                shrinkB=2,
                                connectionstyle="arc3,rad=-0.18"))

    ax.axvline(reach_x, color=C_REACH, linewidth=1.3, linestyle=":", zorder=4,
               label="reach limit  $x=%.3f$" % reach_x)
    ax.text(reach_x - 0.006, plat_c[1] - plat_h[1] + 0.02,
            "max_base_x + grasp_max", rotation=90, fontsize=7, color=C_REACH,
            ha="right", va="bottom", zorder=7)

    ax.set_xlabel("world $x$ (m)  —  robot approaches from $-x$")
    ax.set_ylabel("world $y$ (m)")
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    pad = 0.05
    ax.set_xlim(plat_c[0] - plat_h[0] - pad, max(reach_x, plat_c[0] + plat_h[0]) + pad)
    ax.set_ylim(plat_c[1] - plat_h[1] - pad, plat_c[1] + plat_h[1] + pad)

    # a metric scale bar, so the figure is readable if it is ever cropped
    bx = plat_c[0] - plat_h[0] + 0.01
    by = plat_c[1] - plat_h[1] - pad + 0.018
    ax.plot([bx, bx + 0.10], [by, by], color="k", lw=2.0, solid_capstyle="butt",
            zorder=8)
    ax.text(bx + 0.05, by + 0.008, "0.10 m", fontsize=7, ha="center", zorder=8)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=7.5, handlelength=1.6, borderaxespad=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, "platform_regions")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- the caption, written next to the figure so they cannot drift --------
    caption = f"""Figure 3.x — Box spawn regions on the pickup platform (world XY).

The platform footprint ({2 * plat_h[0]:.2f} x {2 * plat_h[1]:.2f} m) is read from
the model geom `{box.platform_geom}`; the box footprint
({2 * box_h[0]:.2f} x {2 * box_h[1]:.2f} m) from `{box.box_geom}`. Box position is
the ONLY randomised quantity — both platforms are fixed (Q6), so Objective 4
measures generalization over manipulation targets and never over navigation
targets. The sample region spans x [{reg_c[0] - reg_h[0]:.2f}, {reg_c[0] + reg_h[0]:.2f}] m
and y [{reg_c[1] - reg_h[1]:.2f}, {reg_c[1] + reg_h[1]:.2f}] m. Its x extent is set by the
ROBOT, not by the platform: the walk-in drives the base to (box_x - standoff),
so the far sample edge must leave the base clear of max_base_x = {grasp.max_base_x:.3f} m,
which trims x to +/-{reg_h[0]:.2f} m (O22) and leaves {plat_h[0] - reg_h[0] - box_h[0]:.2f} m of platform unused on
each side. The dotted line at x = {reach_x:.3f} m is the separate, looser ARM-reach
bound ({grasp.max_base_x:.3f} + {grasp.grasp_max:.2f}); the standoff constraint binds first.
The y extent is limited only by the platform edge, with the box footprint at a
corner spawn clearing it by {1000 * gap[1]:.0f} mm. {N_SEEDS} draws through the real sampler
are overlaid, {int(held.sum())} of them ({100 * frac:.1f}%) inside the held-out patch,
against {100 * predicted:.1f}% predicted by area.

The held-out patch is a 2-D INTERIOR patch, and that is the point. Every
held-out x value also occurs in training at some other y, and every held-out y
also occurs in training at some other x — both marginals remain
in-distribution, so only the COMBINATION is unseen. Objective 4 therefore
measures compositional generalization strictly inside the convex hull of the
training data, not extrapolation beyond it. A far-half or edge-band split would
have measured the opposite thing.
"""
    with open(stem + "_caption.txt", "w", encoding="utf-8") as fh:
        fh.write(caption)
    print("\nwrote %s.pdf, %s.png, %s_caption.txt" % (stem, stem, stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
