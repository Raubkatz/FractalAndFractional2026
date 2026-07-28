import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


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
TRIANGLE_COLOR = PALETTE[6]   # Sierpinski triangle color from project palette


# ============================================================
# OUTPUT SETTINGS
# ============================================================
OUTPUT_DIR = "sierpinski_triangle_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FRAME SETTINGS
# ============================================================
ITERATIONS = 10                # save individual frames for depths 1,...,10
LINE_WIDTH = 0.25
FIGSIZE = (6, 6)
DPI = 220
TRANSPARENT_BG = True          # True = transparent PNG background


# ============================================================
# SIERPINSKI TRIANGLE GEOMETRY
# ============================================================
def midpoint(p1, p2):
    """Midpoint between two 2D points."""
    return ((p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0)


def sierpinski_subdivide(triangle):
    """
    Replace one filled triangle by the 3 remaining Sierpinski triangles.
    The central triangle is removed.
    """
    a, b, c = triangle

    ab = midpoint(a, b)
    bc = midpoint(b, c)
    ca = midpoint(c, a)

    return [
        (a, ab, ca),
        (ab, b, bc),
        (ca, bc, c),
    ]


def refine_triangles(triangles):
    """Apply one Sierpinski refinement step to all current triangles."""
    new_triangles = []
    for triangle in triangles:
        new_triangles.extend(sierpinski_subdivide(triangle))
    return new_triangles


def build_levels(iterations):
    """
    Create list of triangle levels.
    level 0 = filled triangle
    level n = Sierpinski refinement of previous level
    """
    levels = []

    height = math.sqrt(3.0) / 2.0
    base = [
        ((0.0, 0.0), (1.0, 0.0), (0.5, height))
    ]
    levels.append(base)

    current = base
    for _ in range(iterations):
        current = refine_triangles(current)
        levels.append(current)

    return levels


# ============================================================
# PLOTTING
# ============================================================
def get_bounds(levels, pad=0.06):
    xs = []
    ys = []
    for level in levels:
        for triangle in level:
            for x, y in triangle:
                xs.append(x)
                ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = x_max - x_min
    dy = y_max - y_min

    return (
        x_min - pad * max(dx, dy),
        x_max + pad * max(dx, dy),
        y_min - pad * max(dx, dy),
        y_max + pad * max(dx, dy),
    )


def render_frame(triangles, bounds, output_stem):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    patches = [
        Polygon(triangle, closed=True)
        for triangle in triangles
    ]

    collection = PatchCollection(
        patches,
        facecolor=TRIANGLE_COLOR,
        edgecolor=TRIANGLE_COLOR,
        linewidth=LINE_WIDTH,
        antialiased=True,
    )
    ax.add_collection(collection)

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
def create_sierpinski_frames(iterations=ITERATIONS):
    levels = build_levels(iterations)
    bounds = get_bounds(levels)

    for level_idx in range(1, iterations + 1):
        output_stem = f"sierpinski_triangle_depth_{level_idx:02d}"
        render_frame(levels[level_idx], bounds, output_stem)

    print(f"Saved {iterations} Sierpinski triangle depths as PNG and EPS.")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    create_sierpinski_frames()