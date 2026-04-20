---
type: sensei_note
task: 117
tags: [transformation, symmetry-reflection, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

## The rule

This task involves two colored shapes on a black background. One shape (typically forming a symmetric diamond or cross pattern) acts as the **reflection center**. The other shape is **reflected across both axes** passing through the center shape's bounding box center.

Specifically:
1. Identify the two non-zero colors in the grid
2. Determine which shape is the "center" shape (the one with 4-way rotational/reflection symmetry around its center)
3. Calculate the center point of the center shape's bounding box
4. Keep the center shape unchanged
5. For the other shape, create up to 4 copies: the original position, vertically reflected, horizontally reflected, and both reflected across the center point

The reflection formula for a point (r, c) across center (cr, cc) is:
- Vertical reflection: (2×cr - r, c)
- Horizontal reflection: (r, 2×cc - c)
- Both: (2×cr - r, 2×cc - c)

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
        # Fallback: use the shape with fewer pixels
        center_color = min(colors, key=lambda c: len(pos[c]))
    
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

This solution uses the **symmetry-reflection** primitive family. The key insight is recognizing that one shape serves as an anchor point (identified by its intrinsic 4-way symmetry), while the other shape is the "content" to be mirrored. This pattern appears in multiple ARC tasks where objects need to be reflected around a central point or axis.

The algorithm generalizes because:
1. It doesn't hardcode specific colors - it detects whichever shape has symmetry
2. It works for any grid size and any positions of the shapes
3. The reflection math (2×center - position) is universal for point reflection
4. Boundary checking ensures reflections stay within the grid

This is a fundamental geometric transformation pattern that Erebus should recognize: **identify anchor → compute center → reflect content across axes**.
