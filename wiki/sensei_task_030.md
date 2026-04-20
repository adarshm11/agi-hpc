---
type: sensei_note
task: 30
tags: [transformation, vertical-alignment, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

# Task 030: Vertical Alignment to Color-1 Anchor

## The rule

This task involves **vertical alignment** of multiple colored objects. The rule is:

1. Identify all colored objects in the grid (each color forms one or more connected components, but we treat all cells of the same color as one object for alignment purposes).
2. Find the **top row** (minimum row index) of the color-1 object.
3. Shift every other colored object **vertically** so that their top rows align with the top row of the color-1 object.
4. **Horizontal positions remain unchanged** — only vertical shifting occurs.
5. The color-1 object itself does not move (it serves as the anchor).

The key insight: **color-1 is the anchor**. All other colors move to match its vertical position.

## Reference implementation

```python
def transform(grid):
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    
    # Collect cells by color
    colors = {}
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0:
                if val not in colors:
                    colors[val] = []
                colors[val].append((r, c))
    
    # If no 1s exist, return empty grid
    if 1 not in colors:
        return [[0] * w for _ in range(h)]
    
    # Find the top row of the 1s object
    ones_top = min(r for r, c in colors[1])
    
    # Create output grid
    output = [[0] * w for _ in range(h)]
    
    # For each color, shift vertically to align with 1s top
    for color, cells in colors.items():
        color_top = min(r for r, c in cells)
        shift = ones_top - color_top
        
        for r, c in cells:
            new_r = r + shift
            if 0 <= new_r < h:
                output[new_r][c] = color
    
    return output
```

## Why this generalizes

This belongs to the **vertical-alignment** primitive family. The pattern is:

- **Anchor identification**: One color (here, color-1) serves as the reference point.
- **Relative transformation**: All other objects transform relative to the anchor's position.
- **Preservation of shape**: Objects maintain their internal structure; only their absolute position changes.
- **Single-axis transformation**: Only vertical (row) coordinates change; horizontal (column) coordinates are preserved.

This generalizes to any task where objects need to align to a reference object along a single axis. The anchor color could vary by task, but the mechanism (find anchor position, compute shift, apply to all objects) remains the same.
