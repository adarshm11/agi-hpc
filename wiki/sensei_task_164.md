---
type: sensei_note
task: 164
tags: [expansion, horizontal-mirror, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

# Task 164: Horizontal Mirror Expansion

## The rule

The transformation creates a horizontal mirror image of each row. For every row in the input grid, concatenate the row with its reverse. This doubles the width of the grid while keeping the height the same. The result is that each row becomes palindromic (symmetric left-to-right).

For example, a row `[6, 8, 1]` becomes `[6, 8, 1, 1, 8, 6]` - the original three elements followed by those same three elements in reverse order.

## Reference implementation

```python
def transform(grid):
    result = []
    for row in grid:
        result.append(row + row[::-1])
    return result
```

## Why this generalizes

This belongs to the **horizontal-mirror** primitive family, a common pattern in ARC where symmetry operations are applied to expand grids. The key insight is recognizing that the output width is exactly 2× the input width, and examining any single row reveals the mirroring pattern: the right half is always the reverse of the left half. This pattern holds regardless of the specific colors/values in the grid, making it a robust geometric transformation that generalizes to any input dimensions.
