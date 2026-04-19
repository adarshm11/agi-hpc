---
type: sensei_note
task: 381
tags: [objects, rectangle-pairs, gap-fill, blocked-pairs, arc]
written_by: Professor Bond
written_at: 2026-04-19
verified_by: reference_implementation (train 3/3, test 1/1)
---

# Task 381 — Fill Gap Between Rectangle Pairs With Color 9

## Object-level reasoning, not pixel-level.

Operate on connected components of the marker color (2s). Previous
sensei note was wrong about the rule — do not trust earlier attempts.

## Verified rule

1. Find connected components of 2s. Compute each bounding box
   `(r0, r1, c0, c1)`.
2. For every ordered pair of rectangles `(A, B)`:
   a. Compute row-overlap: `rs = max(A.r0, B.r0)`,
      `re = min(A.r1, B.r1)`. Skip if `rs > re`.
   b. Compute the strict horizontal gap: if `A` is left of `B`
      (`A.c1 < B.c0`), gap cols are `A.c1+1 .. B.c0-1`. Symmetrically
      if `B` is left of `A`. Skip if they overlap horizontally.
   c. **Blocked-pair check** — skip the ENTIRE pair if any third
      rectangle `C` satisfies:
        - `C`'s row range intersects `[rs, re]`, AND
        - `C`'s col range intersects the gap `[left+1, right-1]`.
   d. Otherwise, for every `(r, c)` with `rs <= r <= re` and
      `left+1 <= c <= right-1`, set the output cell to **9** (not 2).
      Leave non-zero cells alone (defensive — in practice they are 0).
3. Pixels outside filled gaps are copied from the input unchanged.

## Key points that tripped earlier attempts

- **Fill color is 9, not 2.** "Fill with marker color" is wrong.
- **No width/height matching.** Pairs fill whenever they row-overlap
  and are horizontally separated, regardless of size.
- **Blocking matters.** In train ex2, rectangles E (rows 7-9, cols
  0-3) and D (rows 6-9, col 9) row-overlap on rows 7-9, but rect C
  (rows 3-8, cols 5-7) blocks rows 7-8. The entire (E, D) pair is
  skipped, including row 9 which looks free. The fill is delivered
  via the E-C and C-D chain instead.
- **Horizontal gaps only in this task.** No vertical-gap fills
  appear in the 3 training examples, so the reference solution only
  handles horizontal. Keep the code symmetric if you extend it, but
  the test set never exercises vertical.

## Reference implementation

```python
import numpy as np
from collections import deque

def _components(mask):
    m = np.asarray(mask); H, W = m.shape
    seen = np.zeros_like(m, dtype=bool); rects = []
    for i in range(H):
        for j in range(W):
            if not m[i, j] or seen[i, j]: continue
            q = deque([(i, j)]); seen[i, j] = True
            r0 = r1 = i; c0 = c1 = j
            while q:
                r, c = q.popleft()
                r0 = min(r0, r); r1 = max(r1, r)
                c0 = min(c0, c); c1 = max(c1, c)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True; q.append((nr, nc))
            rects.append((r0, r1, c0, c1))
    return rects

def transform(grid):
    g = np.array(grid); out = g.copy()
    rects = _components(g == 2); n = len(rects)
    for i in range(n):
        for j in range(i+1, n):
            r0a, r1a, c0a, c1a = rects[i]
            r0b, r1b, c0b, c1b = rects[j]
            rs, re = max(r0a, r0b), min(r1a, r1b)
            if rs > re: continue
            if c1a < c0b: L, R = c1a, c0b
            elif c1b < c0a: L, R = c1b, c0a
            else: continue
            blocked = False
            for k in range(n):
                if k in (i, j): continue
                r0c, r1c, c0c, c1c = rects[k]
                if max(rs, r0c) > min(re, r1c): continue
                if c1c < L + 1 or c0c > R - 1: continue
                blocked = True; break
            if blocked: continue
            for r in range(rs, re + 1):
                for c in range(L + 1, R):
                    if out[r, c] == 0: out[r, c] = 9
    return out.tolist()
```

Verified against `train[0..2]` and `test[0]` — all pass exactly.
