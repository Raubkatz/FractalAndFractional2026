import math
import os

import matplotlib.pyplot as plt


# ============================================================
# COLOR SETTINGS
# ============================================================
PALETTE = [
    "#589641",   # 0  green
    "#367DCD",   # 1  blue
    "#874BD2",   # 2  purple
    "#DC6923",   # 3  orange
    "#E1AA14",   # 4  gold
    "#2896A0",   # 5  teal
    "#BE372D",   # 6  red
    "#2F7F74",   # 7  dark teal
    "#C05A8A",   # 8  pink
]

BG_COLOR = "#FFFFFF"      # background
LINE_COLOR = PALETTE[6]   # Koch curve color from project palette


# ============================================================
# OUTPUT SETTINGS
# ============================================================
OUTPUT_DIR = "koch_curve_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FRAME SETTINGS
# ============================================================
ITERATIONS = 10                # save individual frames for depths 1,...,10
LINE_WIDTH = 3.0
FIGSIZE = (12, 4)
DPI = 220
TRANSPARENT_BG = True          # True = transparent PNG background


# ============================================================
# KOCH CURVE GEOMETRY
# ============================================================
def interpolate(p1, p2, t):
    """Linear interpolation between two 2D points."""
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)


def koch_subdivide(p1, p2):
    """
    Replace one line segment by the 4 Koch segments.
    Returns 5 points: A, B, C, D, E
    where segment A->E becomes A->B->C->D->E.
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # points at 1/3 and 2/3
    b = (x1 + dx / 3.0, y1 + dy / 3.0)
    d = (x1 + 2.0 * dx / 3.0, y1 + 2.0 * dy / 3.0)

    # rotate vector (d - b) by +60° around b to get peak c
    angle = math.radians(60)
    vx = d[0] - b[0]
    vy = d[1] - b[1]

    cx = b[0] + vx * math.cos(angle) - vy * math.sin(angle)
    cy = b[1] + vx * math.sin(angle) + vy * math.cos(angle)
    c = (cx, cy)

    return [p1, b, c, d, p2]


def refine_polyline(points):
    """Apply one Koch refinement step to the whole polyline."""
    new_points = []
    for i in range(len(points) - 1):
        sub = koch_subdivide(points[i], points[i + 1])
        if i == 0:
            new_points.extend(sub)
        else:
            new_points.extend(sub[1:])  # avoid duplicate point
    return new_points


def build_levels(iterations):
    """
    Create list of polyline levels.
    level 0 = straight line
    level n = Koch refinement of previous level
    """
    levels = []
    base = [(0.0, 0.0), (1.0, 0.0)]
    levels.append(base)

    current = base
    for _ in range(iterations):
        current = refine_polyline(current)
        levels.append(current)

    return levels


# ============================================================
# PLOTTING
# ============================================================
def get_bounds(levels, pad=0.06):
    xs = []
    ys = []
    for level in levels:
        for x, y in level:
            xs.append(x)
            ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = x_max - x_min
    dy = y_max - y_min

    return (
        x_min - pad * dx,
        x_max + pad * dx,
        y_min - pad * max(dx, dy),
        y_max + pad * max(dx, dy),
    )


def render_frame(points, bounds, output_stem):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.plot(
        xs,
        ys,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
    )

    # Minimal clean style
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # Optional subtle frame disabled
    for spine in ax.spines.values():
        spine.set_visible(False)

    png_path = os.path.join(OUTPUT_DIR, f"{output_stem}.png")
    eps_path = os.path.join(OUTPUT_DIR, f"{output_stem}.eps")

    plt.savefig(
        png_path,
        format="png",
        dpi=DPI,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=TRANSPARENT_BG,
    )

    plt.savefig(
        eps_path,
        format="eps",
        facecolor=BG_COLOR,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    print(f"PNG saved to: {os.path.abspath(png_path)}")
    print(f"EPS saved to: {os.path.abspath(eps_path)}")


# ============================================================
# MAIN FRAME CREATION
# ============================================================
def create_koch_frames(iterations=ITERATIONS):
    levels = build_levels(iterations)
    bounds = get_bounds(levels)

    for level_idx in range(1, iterations + 1):
        output_stem = f"koch_curve_depth_{level_idx:02d}"
        render_frame(levels[level_idx], bounds, output_stem)

    print(f"Saved {iterations} Koch curve depths as PNG and EPS.")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    create_koch_frames()