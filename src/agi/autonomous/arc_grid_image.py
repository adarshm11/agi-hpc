"""Render ARC grids as PNG bytes for multimodal LLM consumption.

ARC tasks use a 10-color palette (0-9). Each cell becomes a solid-color
tile with a thin grid line between tiles so boundaries are visible.
"""

from __future__ import annotations

import io

# ARC standard palette (RGB)
ARC_COLORS = {
    0: (0, 0, 0),  # black (background)
    1: (0, 116, 217),  # blue
    2: (255, 65, 54),  # red
    3: (46, 204, 64),  # green
    4: (255, 220, 0),  # yellow
    5: (170, 170, 170),  # gray
    6: (240, 18, 190),  # magenta
    7: (255, 133, 27),  # orange
    8: (127, 219, 255),  # azure
    9: (135, 12, 37),  # maroon
}

GRID_LINE = (60, 60, 70)


def grid_to_png(grid: list[list[int]], cell_px: int = 32, line_px: int = 1) -> bytes:
    """Render one ARC grid as a PNG. Returns raw bytes.

    ``cell_px`` is the size of each cell in pixels. ``line_px`` is the
    thickness of the gridline between cells (0 to disable).
    """
    from PIL import Image, ImageDraw

    if not grid or not grid[0]:
        img = Image.new("RGB", (cell_px, cell_px), ARC_COLORS[0])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    h = len(grid)
    w = len(grid[0])
    img = Image.new("RGB", (w * cell_px, h * cell_px), GRID_LINE)
    draw = ImageDraw.Draw(img)

    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            color = ARC_COLORS.get(int(v), ARC_COLORS[0])
            x0 = c * cell_px + line_px
            y0 = r * cell_px + line_px
            x1 = (c + 1) * cell_px - line_px
            y1 = (r + 1) * cell_px - line_px
            draw.rectangle([x0, y0, x1, y1], fill=color)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def example_to_pair_png(example: dict, cell_px: int = 32, gap_px: int = 16) -> bytes:
    """Render one train/test example (input ➔ output) as a single PNG.

    Useful for showing the multimodal LLM the full transform visually.
    """
    from PIL import Image

    inp = example.get("input") or [[0]]
    out = example.get("output") or [[0]]
    left = Image.open(io.BytesIO(grid_to_png(inp, cell_px=cell_px)))
    right = Image.open(io.BytesIO(grid_to_png(out, cell_px=cell_px)))

    w = left.width + gap_px + right.width
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (w, h), (20, 20, 28))
    canvas.paste(left, (0, (h - left.height) // 2))
    canvas.paste(right, (left.width + gap_px, (h - right.height) // 2))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def task_to_pngs(task: dict, max_examples: int = 3, cell_px: int = 32) -> list[bytes]:
    """Render up to ``max_examples`` training examples as side-by-side
    input→output PNGs. Returns list of PNG bytes."""
    return [
        example_to_pair_png(ex, cell_px=cell_px)
        for ex in task.get("train", [])[:max_examples]
    ]
