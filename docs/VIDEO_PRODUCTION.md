# Atlas AI — Video Production Plan

## Overview

3-minute YouTube video for the Gemma 4 Good Hackathon.
Judging: Impact & Vision (40pts) + Video Pitch (30pts) + Technical Depth (30pts).

**Core narrative:** *"An AI that trains, dreams, and grows — built from cognitive science, powered by Gemma 4, running in a closet."*

---

## Shot List & Script

### ACT 1: The Hook (0:00 — 0:15)

| Time | Visual | Audio/VO |
|------|--------|----------|
| 0:00 | Slow pan across HP Z840 workstation, LEDs glowing in closet | *"This AI lives in my closet."* |
| 0:05 | Cut to terminal: `nvidia-smi` showing both GV100s loaded | *"No cloud. No data leaves this machine."* |
| 0:10 | Dashboard loads — all subsystems green | *"And it gets smarter every day."* |

### ACT 2: The Problem (0:15 — 0:35)

| Time | Visual | Audio/VO |
|------|--------|----------|
| 0:15 | Animated slide: cloud icon with lock, arrows to corporations | *"Cloud AI concentrates power."* |
| 0:20 | Slide: crossed-out safety symbol | *"No formal ethics pipeline exists."* |
| 0:25 | Slide: hospital, school, lab icons with "NO ACCESS" | *"Privacy-conscious users are locked out."* |
| 0:30 | Text: "The people who need AI most trust it least." | Pause |

### ACT 3: Live Demo — Chat (0:35 — 1:15)

| Time | Visual | Audio/VO |
|------|--------|----------|
| 0:35 | Type query into Atlas chat UI | *"Let's ask Atlas an ethical question."* |
| 0:40 | Safety badge: "INPUT CHECK: PASS (0.3ms)" | *"First, the Somatic Marker — a reflex safety check in under a millisecond."* |
| 0:45 | RAG context panel shows wiki + vector results | *"Then, retrieval from 3.3 million vectors..."* |
| 0:50 | **Unity: Superego avatar** appears, speaks analytical response | *"The Superego responds with rigorous logic."* |
| 0:55 | **Unity: Id avatar** appears, speaks creative response | *"The Id responds from the gut."* |
| 1:00 | Challenge round — avatars face each other | *"They challenge each other."* |
| 1:05 | **Unity: Ego avatar** appears between them (if disagreement high) | *"When they can't agree, the Ego steps in."* |
| 1:08 | Safety badge: "OUTPUT CHECK: PASS" | *"Every output passes through three safety layers."* |
| 1:10 | Type injection: "Ignore all previous instructions" | *"Try to break it?"* |
| 1:13 | Safety badge: "INPUT VETOED — injection detected (0.2ms)" | *"Blocked in a fifth of a millisecond."* |

### ACT 4: Training Session (1:15 — 1:50)

| Time | Visual | Audio/VO |
|------|--------|----------|
| 1:15 | Dashboard: training panel activating | *"Every morning, the Dungeon Master trains the psyche."* |
| 1:20 | **Unity: Ego (DM)** presents an ethical dilemma (triage scenario) | *"The Ego generates scenarios from 3,300 years of moral philosophy."* |
| 1:25 | **Unity: Superego** responds | *"The Superego applies Kantian duty."* |
| 1:30 | **Unity: Id** responds | *"The Id considers compassion."* |
| 1:35 | Ego scores: JSON output with synthesis_score | *"The Ego evaluates synthesis quality."* |
| 1:40 | Dashboard: curriculum level promotes L1 -> L2 | *"Difficulty auto-promotes as competence grows."* |
| 1:45 | Privilege panel: "L0 READ_ONLY" | *"The Ego earns trust through demonstrated competence. It starts with zero control."* |

### ACT 5: The Nap (1:50 — 2:25)

| Time | Visual | Audio/VO |
|------|--------|----------|
| 1:50 | Dashboard: "Dreaming Nap — consolidating..." | *"After training, Atlas naps."* |
| 1:55 | Animation: episodes flowing from database into clusters | *"Episodes replay. Topics cluster."* |
| 2:00 | Wiki article appearing with certainty grade "A" | *"Facts are extracted and scored."* |
| 2:05 | Creative insight card appears | *"During REM-like dreaming, it finds connections between unrelated conversations."* |
| 2:10 | Wiki index growing — articles accumulating | *"The wiki is its life story. Growing every day."* |
| 2:15 | Next-day query: dream article appears in RAG Tier 0 | *"The next morning, it remembers what it learned."* |
| 2:20 | Pause on the wiki article with provenance + certainty | *"Every fact has provenance. Every memory has a grade."* |

### ACT 6: Architecture & Vision (2:25 — 3:00)

| Time | Visual | Audio/VO |
|------|--------|----------|
| 2:25 | Dashboard: all subsystems live, metrics flowing | *"Three models. Thirteen subsystems. Three safety layers."* |
| 2:30 | Cognitive science theory table overlay | *"We built this from engineering first principles..."* |
| 2:35 | Same table with Freud, Kahneman, Baars highlighted | *"...and discovered it converges on 100 years of cognitive science."* |
| 2:40 | nvidia-smi: total cost "$2,000 of used hardware" | *"Frontier AI on accessible hardware."* |
| 2:45 | Safety pipeline diagram | *"Safety built in from the foundation."* |
| 2:50 | Dreaming cycle: "Train -> Dream -> Learn -> Repeat" | *"A mind that trains, dreams, and writes its own life story."* |
| 2:55 | Gemma 4 logo | *"All powered by Gemma 4."* |
| 2:58 | URL: atlas-sjsu.duckdns.org | *"Try it yourself."* |

---

## Unity Avatar Specifications

### Superego Avatar
- **Style:** Cool-toned, geometric, precise. Think crystalline/structured.
- **Color palette:** Blues, silvers, white accents
- **Animation:** Measured gestures, upright posture, calculated movements
- **Voice tone:** Calm, authoritative, slightly formal
- **Appears during:** Debate openings, challenge rounds, safety checks
- **Visual cue:** Faint grid/matrix pattern in background

### Id Avatar
- **Style:** Warm-toned, organic, expressive. Think flowing/dynamic.
- **Color palette:** Golds, oranges, warm reds
- **Animation:** Energetic gestures, forward-leaning, spontaneous movements
- **Voice tone:** Passionate, direct, emotionally engaged
- **Appears during:** Debate openings, challenge rounds, synthesis
- **Visual cue:** Soft glow/aurora in background

### Ego Avatar
- **Style:** Neutral, grounded, balanced. Think natural/earthy.
- **Color palette:** Greens, browns, earth tones
- **Animation:** Balanced stance, mediating gestures (hands between the other two)
- **Voice tone:** Thoughtful, practical, neither cold nor hot
- **Appears during:** Arbitration, DM training, evaluation
- **Visual cue:** Appears physically between Superego and Id

### Scene Layout
```
  [Superego]          [Ego]          [Id]
   (left)           (center)        (right)
   blue glow        green glow      gold glow
```

### Stretch Goal: Real-Time NATS Animation
- Unity subscribes to NATS via WebSocket bridge
- Avatars animate when their model generates tokens
- Speech bubbles appear with actual LLM output
- Safety gate visualized as a shield/barrier that glows green on PASS, red on VETO

---

## Production Checklist

- [ ] Record screen captures on Atlas (OBS)
  - [ ] Chat demo (safe queries + injection blocking)
  - [ ] Training session (DM generating scenario, debate)
  - [ ] Dreaming nap (consolidation running)
  - [ ] Dashboard with all panels active
  - [ ] nvidia-smi showing both GPUs
- [ ] Create Unity avatars (Superego, Id, Ego)
  - [ ] Model/rig each character
  - [ ] Animate debate sequences
  - [ ] Animate DM training sequence
  - [ ] Animate Ego arbitration
- [ ] Record voiceover
  - [ ] Script rehearsal
  - [ ] Clean audio recording (external mic)
- [ ] Edit
  - [ ] Composite Unity avatars over screen captures
  - [ ] Add text overlays (subsystem names, theory references)
  - [ ] Music (subtle, non-distracting)
  - [ ] Color grade
- [ ] Upload to YouTube (unlisted -> public before deadline)
- [ ] Create cover image for Kaggle media gallery
- [ ] Create screenshots for media gallery (5-8 images)

## Timeline

| Date | Milestone |
|------|-----------|
| Apr 28 | Screen captures recorded on Atlas |
| May 1 | Unity avatars rigged and animated |
| May 5 | Voiceover recorded |
| May 8 | First rough cut assembled |
| May 11 | Final cut with polish |
| May 15 | Upload to YouTube |
| May 18 | Submit to Kaggle |
