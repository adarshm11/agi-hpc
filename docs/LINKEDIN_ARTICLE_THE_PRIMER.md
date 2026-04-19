# The Primer: giving an autonomous ARC solver a teacher

Erebus is an autonomous ARC puzzle solver we run on Atlas. It works through Kaggle's NeuroGolf task set by generating candidate Python programs, running them against each training pair, scoring itself, updating a memory file, and retrying with a different strategy. It was designed to be self-directed.

It turns out "self-directed" is not the same as "self-improving". A week into running it, Erebus had accumulated 50+ attempts on several tasks without making real progress. It wasn't asking anyone for help. It was just spinning on the same wrong hypotheses. One morning I added a `_ask_for_help` tool so it could at least surface its stuck state, and by the evening it had posted entries like this to a JSON file:

```
task381: I have tried 57 times (best: 2/3). Error types:
  reasoning, execution, perception. I need guidance: is this
  transformation local or global? Am I missing a spatial primitive?
```

Nobody was reading the file.

That is how the Primer got built. It is a small always-on daemon living at `src/agi/primer/service.py`, roughly 600 lines. Its only job is to read the help queue and write useful replies into a wiki that Erebus reads.

## The naive version actively hurt

The obvious implementation is: poll the help queue every ten minutes, pick a stuck task, send it to the smartest LLM you have, write the answer to a wiki note. I had this running for about three hours before I looked at the output.

Kimi (running on our NRP cluster) had been asked about task 381. It returned a confident rule. The rule was wrong in two distinct ways, but it sounded plausible enough that our commit pipeline published it. Erebus picked up the note on its next tick. It started applying the wrong rule. Because the rule appeared consistent with its existing memory of the task, its internal "does this attempt make sense" check passed. The new failing attempts were logged as real failures, not as "I followed the sensei note and it didn't work."

By the time I noticed, Erebus had 102 failed attempts on that one task, most of them variations of the wrong rule that the wiki told it was correct.

A wrong sensei note is worse than no sensei note. It actively entrenches a bad hypothesis and costs you the investigation that the agent would have done on its own.

## Verify, then publish

The actual Primer works differently. When it picks up a stuck task, it consults the vMOE ensemble (Kimi, GLM-4.7, and Qwen3 on NRP) and asks each one for a candidate `transform(grid) -> grid` function. It does not publish their answer. Instead it hands the candidate to a validator.

The validator (`src/agi/primer/validator.py`) runs the candidate in an isolated subprocess with a 10-second timeout. It loads the task file, iterates over every training example, runs the candidate, compares the output byte-for-byte with the expected output, and then does the same for the test example. Only if all comparisons match does the candidate make it through. The publisher then extracts the rule into prose and writes a sensei note that includes the verified reference implementation.

In other words, the LLM proposes and a deterministic oracle disposes. The bottleneck is the oracle, not the LLM.

This is roughly 60 lines of code and it is the piece that makes the whole system trustworthy.

## The loop

```
tick():
  stuck_tasks = read help queue, apply cooldown filter
  for task in stuck_tasks[:3]:
      for expert in vmoe.experts (first_verified policy):
          candidate = expert.propose(task)
          if validator.verify(candidate, task):
              publish_sensei_note(task, candidate)
              break
      else:
          set_cooldown(task, 6h)
```

The vMOE (short for "virtual mixture of experts") is a small abstraction over the three NRP-hosted LLMs. It supports four policies: route by task features, cascade cheap-to-expensive, ensemble with quorum voting, or first_verified. Production runs the last one, because it gives you the fastest useful answer without burning the ensemble budget on tasks the cheapest expert can handle.

## Cooldown

An obvious failure mode: if the Primer tries task 381 and no expert produces a valid candidate, and it tries the same task again ten minutes later with the same ensemble, it will waste the same budget on the same unanswerable question. Cooldown state lives in `/archive/neurogolf/primer_cooldown.json`, six hours by default, overridable via `PRIMER_COOLDOWN_S`. When a task gets solved (either by the Primer publishing a verified note, or by Erebus figuring it out on its own), the cooldown entry is cleared so the task is eligible for re-teaching in case Erebus forgets later.

There is a subtle gotcha here. The cooldown file used to be written with `path.write_text(json.dumps(state))`, a two-syscall sequence. A kernel crash or a power event between truncate and write leaves the file empty. The readers were doing `try: json.loads(...) except Exception: pass`, so a corrupted file silently produced an empty cooldown dict, and the Primer would cheerfully re-ask every task on its next tick. I didn't notice for about a week. The fix (atomic writes via `tempfile + fsync + os.replace`) is now in `src/agi/common/atomic_write.py` and covers four other state files that had the same bug.

## What actually unblocked task 381

This afternoon I ran the existing wiki note for task 381 through the validator. It failed on all three training examples. The note had been written months ago, by hand, before the Primer existed. It said something like: "identify pairs of rectangles where widths match AND aligned vertically, OR heights match AND aligned horizontally, then fill the gap between them with the marker color." That is not the rule.

The actual rule for task 381 is: for any two rectangles of 2s whose row ranges overlap and which are horizontally separated, fill the gap with color 9 (not the marker color), unless a third rectangle intersects both the overlap rows and the gap columns. That third-rectangle cancellation is what makes it interesting, because it means a relationship between two objects can be erased by the presence of an unrelated third object.

I wrote a reference implementation, ran it against all three training examples plus the test, got exact matches, and replaced the sensei note. Erebus's next attempt on task 381 solved it.

The lesson is not "the LLM would have gotten it right." The lesson is that the verify-before-publish invariant applied to the Primer's writes, but not to old human-written notes in the same directory. So I'm adding a pre-commit hook that refuses any `wiki/sensei_task_*.md` unless it contains a reference implementation that passes against the task's training fixtures. Same invariant, now enforced at the commit boundary.

## Things I would do earlier next time

Building the Primer took about two days. If I were starting over:

Start with the validator. The verification oracle should exist before any component that could produce unverified output. I had the validator as an afterthought the first afternoon, and paid for it with three hours of wrong sensei notes.

Emit structured logs from day one. The lifecycle logger we added later would have made the wrong-note bug visible within an hour instead of a day. Right now the Primer emits events like `primer.tick_start`, `primer.candidate_generated`, `primer.validation_passed`, `primer.note_published`. The trends dashboard renders them as a timeline per task.

Write atomically. Every stateful service should use an atomic write helper from the first commit. Retrofitting it later is easy; the hard part is noticing the silent corruption.

## Pointers

- Repo: github.com/ahb-sjsu/agi-hpc
- `src/agi/primer/service.py`, the daemon
- `src/agi/primer/vmoe.py`, the ensemble policy
- `src/agi/primer/validator.py`, the verification oracle
- `src/agi/common/atomic_write.py`, crash-safe state persistence
- `docs/THE_PRIMER.md`, operations reference
- `docs/VMOE.md`, ensemble policy notes

The piece I am still working on: making the Primer proactively scan for inconsistencies between existing sensei notes and the training fixtures, not just respond to Erebus's help queue. That is how we keep a human-authored wrong note from quietly misleading the agent for 102 attempts next time.
