---
type: sensei_note
task: 117
tags: [transformation, symmetry-reflection, arc, primer]
written_by: The Primer
written_at: 2026-04-21
verified_by: run-against-train (all examples pass)
---

## The rule

This task contains two colored shapes on a black (0) background. One shape exhibits **4-way symmetry** (typically a diamond or cross pattern) and acts as the **reflection anchor**. The other shape is **asymmetric** and gets reflected across both axes passing through the anchor shape's center.

**Step-by-step:**
1. Identify the two non-zero colors in the grid
2. For each color, compute the center of its bounding box: `center_row = (min_row + max_row) / 2`, `center_col = (min_col + max_col) / 2`
3. Determine which shape is the anchor by checking for 4-way symmetry: for every pixel at (r, c), verify that (2×cr−r, c), (r, 2×cc−c), and (2×cr−r, 2×cc−c) also exist in the shape
4. Keep the anchor shape unchanged in the output
5. For the asymmetric shape, create 4 copies: the original position, vertical reflection, horizontal reflection, and diagonal reflection across the anchor's center point

**Reflection formulas** (for point (r, c) across center (cr, cc)):
- Vertical: (2×cr − r, c)
- Horizontal: (r, 2×cc − c)
- Both: (2×cr − r, 2×cc − c)

## Reference implementation

```python
import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find all non-zero colors
    colors = []
    for val in range(1, 10):
        if np.any(grid == val):
            colors.append(val)
    
    if len(colors) != 2:
        return grid.tolist()
    
    # Get positions for each color
    def get_positions(color):
        return [(r, c) for r in range(h) for c in range(w) if grid[r, c] == color]
    
    pos = {c: get_positions(c) for c in colors}
    
    # Calculate center of bounding box for each shape
    def get_center(positions):
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        return (min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0
    
    centers = {c: get_center(pos[c]) for c in colors}
    
    # Check if a shape has 4-way symmetry around its center
    def has_four_way_symmetry(positions, center):
        cr, cc = center
        pos_set = set(positions)
        for r, c in positions:
            vr = int(round(2*cr - r))
            hc = int(round(2*cc - c))
            if (vr, c) not in pos_set or (r, hc) not in pos_set or (vr, hc) not in pos_set:
                return False
        return True
    
    # Determine which is the center shape (the symmetric one)
    center_color = None
    for c in colors:
        if has_four_way_symmetry(pos[c], centers[c]):
            center_color = c
            break
    
    if center_color is None:
        # Fallback: use the shape with smaller bounding box
        def bbox_size(positions):
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]
            return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        center_color = min(colors, key=lambda c: bbox_size(pos[c]))
    
    reflect_color = [c for c in colors if c != center_color][0]
    center_cr, center_cc = centers[center_color]
    
    # Create output
    output = np.zeros((h, w), dtype=int)
    
    # Place center shape
    for r, c in pos[center_color]:
        output[r, c] = center_color
    
    # Reflect the other shape across both axes through center
    for r, c in pos[reflect_color]:
        output[r, c] = reflect_color  # Original
        vr = int(round(2 * center_cr - r))
        hc = int(round(2 * center_cc - c))
        if 0 <= vr < h:
            output[vr, c] = reflect_color  # Vertical reflection
        if 0 <= hc < w:
            output[r, hc] = reflect_color  # Horizontal reflection
        if 0 <= vr < h and 0 <= hc < w:
            output[vr, hc] = reflect_color  # Both reflections
    
    return output.tolist()
```

## Why this generalizes

This solution belongs to the **symmetry-reflection** primitive family. The key insight is recognizing that one object serves as an **intrinsic anchor** (identified by its 4-way rotational/reflection symmetry), while the other object is **content to be mirrored**.

This pattern generalizes because:
1. **Anchor identification is intrinsic**: The symmetric shape can be detected algorithmically without hardcoded colors or positions
2. **Reflection is coordinate-based**: Using the bounding box center ensures the transformation works regardless of where shapes appear in the grid
3. **Color-agnostic**: The logic works for any pair of non-zero colors
4. **Size-invariant**: Works for asymmetric shapes of any size or complexity

Similar patterns appear in ARC tasks involving mirroring, tiling, or symmetric completion where one object defines the transformation space for another.
