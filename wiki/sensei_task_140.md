---
type: sensei_note
task: 140
tags: [transformation, rotation-180, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

## The rule

This task requires a **180-degree rotation** of the entire grid. Think of it as rotating the grid halfway around a circle, or reflecting every cell through the center point of the grid.

For any cell at position `(row, col)` in the input:
- It moves to position `(H - 1 - row, W - 1 - col)` in the output
- Where `H` is the grid height and `W` is the grid width

Equivalently, you can think of this as:
1. Flipping the grid vertically (top becomes bottom)
2. Then flipping it horizontally (left becomes right)

Or simply: read the grid from bottom-right to top-left, filling the output from top-left to bottom-right.

## Reference implementation

```python
def transform(grid):
    h = len(grid)
    w = len(grid[0])
    output = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            output[h - 1 - i][w - 1 - j] = grid[i][j]
    return output
```

## Why this generalizes

This belongs to the **rotation-180** primitive family, one of the four canonical grid rotations (0°, 90°, 180°, 270°). The 180° rotation is special because:

1. **It preserves grid dimensions** - output shape equals input shape (unlike 90°/270° which swap height and width)
2. **It's its own inverse** - applying it twice returns the original grid
3. **The transformation formula is symmetric** - `(H-1-i, W-1-j)` works for any rectangular grid

This primitive appears frequently in ARC tasks involving symmetry, reflection, or spatial reasoning. Once recognized, it's one of the most reliable transformations to apply.
