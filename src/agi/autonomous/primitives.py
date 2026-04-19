"""Geometric primitive library for ARC task solving.

Composable visual operations that Erebus can reference in prompts.
Each function works on raw grids (list[list[int]]) so the LLM can
call them directly in generated transforms.

Erebus's request: "I need a seed set of visual operations I can
compose: convex hull, flood fill with boundary conditions, symmetry
detection, pathfinding. Let my diagnostic strategy suggest WHICH
primitive is missing, not just THAT something is missing."

Categories:
  - Object detection: connected components, bounding boxes
  - Spatial: flood fill, pathfinding, convex hull
  - Symmetry: rotational, reflective, translational detection
  - Color: histogram, dominant color, color mapping
  - Geometry: crop, pad, tile, scale, rotate, flip, transpose
"""

from __future__ import annotations
from collections import deque
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Object detection
# ═══════════════════════════════════════════════════════════════


def connected_components(grid, background=0, diagonal=False):
    """Find connected components of non-background cells.

    Returns list of components, each a list of (row, col, color) tuples.
    """
    g = np.array(grid)
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    components = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == background:
                continue
            comp = []
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc, int(g[cr, cc])))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if g[nr, nc] != background:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            components.append(comp)
    return components


def bounding_box(cells):
    """Get bounding box of a list of (row, col, ...) tuples.

    Returns (min_r, min_c, max_r, max_c).
    """
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def extract_object(grid, cells):
    """Extract an object into its own minimal grid."""
    r0, c0, r1, c1 = bounding_box(cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    obj = np.zeros((h, w), dtype=int)
    for r, c, color in cells:
        obj[r - r0, c - c0] = color
    return obj.tolist()


def objects_with_color(grid, color):
    """Find all connected components of a specific color."""
    g = np.array(grid)
    mask = (g == color).astype(int)
    return connected_components(mask.tolist(), background=0)


# ═══════════════════════════════════════════════════════════════
# Spatial operations
# ═══════════════════════════════════════════════════════════════


def flood_fill(grid, start_r, start_c, new_color, boundary=None):
    """Flood fill from (start_r, start_c) with new_color.

    If boundary is set, stop at cells with that color.
    Otherwise, fill all cells matching the start color.
    """
    g = np.array(grid)
    h, w = g.shape
    old_color = g[start_r, start_c]
    if old_color == new_color:
        return g.tolist()
    queue = deque([(start_r, start_c)])
    g[start_r, start_c] = new_color
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if boundary is not None and g[nr, nc] == boundary:
                    continue
                if g[nr, nc] == old_color:
                    g[nr, nc] = new_color
                    queue.append((nr, nc))
    return g.tolist()


def shortest_path(grid, start, end, passable=None):
    """BFS shortest path from start to end on grid.

    passable: set of colors that can be traversed (default: {0}).
    Returns list of (r,c) or empty list if no path.
    """
    if passable is None:
        passable = {0}
    g = np.array(grid)
    h, w = g.shape
    sr, sc = start
    er, ec = end
    visited = set()
    visited.add((sr, sc))
    queue = deque([(sr, sc, [(sr, sc)])])
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == (er, ec):
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and (nr, nc) not in visited
                and (int(g[nr, nc]) in passable or (nr, nc) == (er, ec))
            ):
                visited.add((nr, nc))
                queue.append((nr, nc, path + [(nr, nc)]))
    return []


def convex_hull_mask(cells, h, w):
    """Create a filled convex hull mask for a set of cells.

    Returns h x w binary grid with 1 inside the hull.
    """
    if not cells:
        return np.zeros((h, w), dtype=int).tolist()
    points = np.array([(c[0], c[1]) for c in cells])
    mask = np.zeros((h, w), dtype=int)
    # Simple scanline: for each row, find min/max col of points
    # and fill between them (approximate convex hull)
    for r in range(h):
        row_points = points[points[:, 0] == r]
        if len(row_points) > 0:
            c_min = row_points[:, 1].min()
            c_max = row_points[:, 1].max()
            mask[r, c_min : c_max + 1] = 1
    # Vertical fill: for each col, fill between min/max row
    for c in range(w):
        col_points = points[points[:, 1] == c]
        if len(col_points) > 0:
            r_min = col_points[:, 0].min()
            r_max = col_points[:, 0].max()
            mask[r_min : r_max + 1, c] = 1
    return mask.tolist()


# ═══════════════════════════════════════════════════════════════
# Symmetry detection
# ═══════════════════════════════════════════════════════════════


def detect_symmetry(grid):
    """Detect symmetry type in a grid.

    Returns dict with keys: horizontal, vertical, diagonal,
    rotational_90, rotational_180. Values are booleans.
    """
    g = np.array(grid)
    h, w = g.shape
    result = {
        "horizontal": bool(np.array_equal(g, g[::-1, :])),
        "vertical": bool(np.array_equal(g, g[:, ::-1])),
        "rotational_180": bool(np.array_equal(g, np.rot90(g, 2))),
    }
    if h == w:
        result["diagonal"] = bool(np.array_equal(g, g.T))
        result["rotational_90"] = bool(np.array_equal(g, np.rot90(g)))
    else:
        result["diagonal"] = False
        result["rotational_90"] = False
    return result


def find_repeating_pattern(grid):
    """Detect if grid is a tiling of a smaller pattern.

    Returns (tile_h, tile_w) or None.
    """
    g = np.array(grid)
    h, w = g.shape
    for th in range(1, h // 2 + 1):
        if h % th != 0:
            continue
        for tw in range(1, w // 2 + 1):
            if w % tw != 0:
                continue
            tile = g[:th, :tw]
            match = True
            for r in range(0, h, th):
                for c in range(0, w, tw):
                    if not np.array_equal(g[r : r + th, c : c + tw], tile):
                        match = False
                        break
                if not match:
                    break
            if match and (th < h or tw < w):
                return (th, tw)
    return None


def find_translation(grid1, grid2):
    """Find translation offset (dr, dc) that maps grid1 to grid2.

    Returns (dr, dc) or None.
    """
    g1 = np.array(grid1)
    g2 = np.array(grid2)
    if g1.shape != g2.shape:
        return None
    # Find non-zero positions
    pos1 = set(zip(*np.where(g1 != 0)))
    pos2 = set(zip(*np.where(g2 != 0)))
    if not pos1 or not pos2:
        return None
    # Try offset from first non-zero in each
    r1, c1 = min(pos1)
    r2, c2 = min(pos2)
    dr, dc = r2 - r1, c2 - c1
    # Verify all positions match
    shifted = {(r + dr, c + dc) for r, c in pos1}
    if shifted == pos2:
        # Verify colors match
        for r, c in pos1:
            nr, nc = r + dr, c + dc
            if g1[r, c] != g2[nr, nc]:
                return None
        return (dr, dc)
    return None


# ═══════════════════════════════════════════════════════════════
# Color operations
# ═══════════════════════════════════════════════════════════════


def color_histogram(grid):
    """Count occurrences of each color. Returns {color: count}."""
    g = np.array(grid)
    unique, counts = np.unique(g, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def dominant_color(grid, exclude_background=True):
    """Find the most common color."""
    hist = color_histogram(grid)
    if exclude_background and 0 in hist:
        del hist[0]
    if not hist:
        return 0
    return max(hist, key=hist.get)


def color_map_between(grid_in, grid_out):
    """Infer color mapping from input to output grid.

    Returns {src_color: dst_color} or None if inconsistent.
    """
    g1 = np.array(grid_in)
    g2 = np.array(grid_out)
    if g1.shape != g2.shape:
        return None
    mapping = {}
    for r in range(g1.shape[0]):
        for c in range(g1.shape[1]):
            s, d = int(g1[r, c]), int(g2[r, c])
            if s in mapping and mapping[s] != d:
                return None
            mapping[s] = d
    return mapping


def unique_colors(grid, exclude_background=True):
    """Return set of unique colors in grid."""
    colors = set(np.array(grid).flatten().tolist())
    if exclude_background:
        colors.discard(0)
    return colors


# ═══════════════════════════════════════════════════════════════
# Geometry operations
# ═══════════════════════════════════════════════════════════════


def crop_to_content(grid, background=0):
    """Crop grid to bounding box of non-background content."""
    g = np.array(grid)
    rows = np.any(g != background, axis=1)
    cols = np.any(g != background, axis=0)
    if not rows.any():
        return [[]]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return g[r0 : r1 + 1, c0 : c1 + 1].tolist()


def pad_grid(grid, top=0, bottom=0, left=0, right=0, fill=0):
    """Pad grid with fill color."""
    g = np.array(grid)
    return np.pad(g, ((top, bottom), (left, right)), constant_values=fill).tolist()


def scale_grid(grid, factor_h, factor_w):
    """Scale grid by repeating each pixel."""
    g = np.array(grid)
    return np.repeat(np.repeat(g, factor_h, axis=0), factor_w, axis=1).tolist()


def overlay(base, top, offset_r=0, offset_c=0, transparent=0):
    """Overlay top grid onto base at offset, treating transparent as see-through."""
    b = np.array(base, dtype=int)
    t = np.array(top, dtype=int)
    for r in range(t.shape[0]):
        for c in range(t.shape[1]):
            br, bc = r + offset_r, c + offset_c
            if 0 <= br < b.shape[0] and 0 <= bc < b.shape[1]:
                if t[r, c] != transparent:
                    b[br, bc] = t[r, c]
    return b.tolist()


# ═══════════════════════════════════════════════════════════════
# Primitive catalog (for LLM prompt injection)
# ═══════════════════════════════════════════════════════════════

PRIMITIVE_CATALOG = """Available geometric primitives (import from agi.autonomous.primitives):

OBJECT DETECTION:
  connected_components(grid, background=0, diagonal=False) -> list of components
  bounding_box(cells) -> (min_r, min_c, max_r, max_c)
  extract_object(grid, cells) -> sub-grid
  objects_with_color(grid, color) -> list of components

SPATIAL:
  flood_fill(grid, start_r, start_c, new_color, boundary=None) -> grid
  shortest_path(grid, start, end, passable={0}) -> [(r,c)...]
  convex_hull_mask(cells, h, w) -> binary grid

SYMMETRY:
  detect_symmetry(grid) -> {horizontal, vertical, diagonal, rotational_90/180}
  find_repeating_pattern(grid) -> (tile_h, tile_w) or None
  find_translation(grid1, grid2) -> (dr, dc) or None

COLOR:
  color_histogram(grid) -> {color: count}
  dominant_color(grid, exclude_background=True) -> int
  color_map_between(grid_in, grid_out) -> {src: dst} or None
  unique_colors(grid, exclude_background=True) -> set

GEOMETRY:
  crop_to_content(grid, background=0) -> grid
  pad_grid(grid, top, bottom, left, right, fill=0) -> grid
  scale_grid(grid, factor_h, factor_w) -> grid
  overlay(base, top, offset_r, offset_c, transparent=0) -> grid
"""
