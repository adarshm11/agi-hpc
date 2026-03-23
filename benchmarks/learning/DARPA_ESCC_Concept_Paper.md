# Expedient Subterranean Command and Control (ESCC)

## DARPA Concept Paper — White Paper Submission

**Principal Investigator:** [Your Name], Sr. Member IEEE
**Affiliations:** San Jose State University (SJSU), Sonoma State University (SSU)
**Contact:** [email] | [phone]
**Date:** March 2026

---

## 1. Problem Statement

The proliferation of low-cost autonomous aerial systems (UAS/drones) has fundamentally altered the survivability calculus for military command and control (C2) infrastructure. In current conflicts — notably Ukraine — any concentration of vehicles, antennas, generators, or personnel is rapidly identified through ISR and engaged with loitering munitions or FPV drones within minutes of detection.

Current C2 options present unacceptable tradeoffs:

| Approach | Limitation |
|---|---|
| Mobile C2 vehicles | Visible signatures, constant relocation disrupts planning, vulnerable during setup |
| Existing structures | Known locations, targetable, not always at required positions |
| Permanent deep bunkers | Fixed, extremely costly ($100M+), 2-5 year construction timelines |
| Dispersed/distributed C2 | Degrades coordination, increases communications complexity |

**There is no current capability to rapidly construct hardened, deep underground C2 facilities at operationally relevant timescales (days, not months).**

## 2. Proposed Concept

ESCC is an integrated engineering system that enables construction of a functional underground command post at 20-30m depth within 5-7 days using a package of 5 military-deployable vehicles.

### 2.1 System Architecture

The system employs a multi-shaft approach:

- **1x Primary Access Shaft (1m diameter)** — personnel access via cable elevator, main equipment lowering
- **1x Emergency Egress Shaft (1m diameter)** — offset 30-50m from primary, redundant access
- **2-3x Utility Shafts (30-50cm diameter)** — ventilation (intake/exhaust), power/comms conduit, water drainage/pumping
- **1x Underground Command Chamber** — excavated laterally at depth, shotcrete-lined

### 2.2 Vehicle Package

| # | Vehicle | Function | Weight Class | Technology Readiness |
|---|---|---|---|---|
| 1 | Primary Drill Rig | 1m shaft boring (auger/rotary) | 25-40t, tracked | TRL 7-8 (adapt existing) |
| 2 | Light Drill Rig | 30-50cm utility boreholes | 10-15t, truck-mounted | TRL 8-9 (COTS) |
| 3 | Concrete Mixer/Pump | Surface mixing, pump down shaft | Standard construction | TRL 9 (COTS) |
| 4 | Support Vehicle | Excavation robot, formwork, mesh, elevator kit | Standard logistics truck | TRL 3-5 (robot is novel) |
| 5 | Generator/Power | Surface power plant | Trailer-mounted | TRL 9 (COTS) |

### 2.3 Construction Sequence

**Phase 1: Site Preparation & Utility Shafts (Day 1)**
- Rapid geological assessment (ground-penetrating radar, soil sampling auger)
- Light rig bores 2-3 utility shafts (hours each at 30-50cm)
- Establish ventilation and dewatering

**Phase 2: Primary Shaft (Days 1-2)**
- Primary rig bores 1m shaft to target depth (20-30m)
- Steel casing or shotcrete lining as bore advances
- Spoil removal (~31m³ at 40m depth, single dump truck load)

**Phase 3: Chamber Excavation (Days 2-5)**
- Deploy foldable excavation robot through primary shaft
- Robot excavates laterally in 1-2m advances
- Shotcrete robot arm lines walls immediately after cutting
- Concrete pumped from surface mixer via utility shaft or dedicated line
- Light rebar mesh positioned by robot or teleoperated placement

**Phase 4: Fitout (Days 5-7)**
- Install cable elevator in primary shaft
- Run power, comms, ventilation ducting through utility shafts
- Install command post equipment (tables, displays, comms racks)
- Bore emergency egress shaft (parallel with fitout)

**Phase 5: Operational**
- Surface footprint reduced to camouflaged shaft heads and offset generator
- Remote/dispersed antenna elements connected via hardline up utility shaft
- Continuous ventilation via dedicated intake/exhaust shafts

## 3. Key Innovation: Foldable Autonomous Excavation Robot

The primary novel technology in ESCC is a **compact, electrically-powered excavation and shotcrete robot** that:

- **Fits through a 1m shaft** in folded transport configuration (<0.9m envelope)
- **Unfolds at depth** into a working configuration for lateral excavation
- **Integrates multiple functions**: cutting head or hydraulic breaker, spoil mucking/conveying, shotcrete nozzle, rebar mesh positioning
- **Powered via surface cable** — no onboard combustion engine, eliminating exhaust and reducing size
- **Teleoperated with autonomous capability** — initial deployment is human-in-the-loop via camera/sensor suite; progressive autonomy as the system matures

### 3.1 Relevant Prior Art

- **Epiroc Scooptram ST7 Battery** — autonomous underground loader (but too large)
- **Sandvik AutoMine** — autonomous underground mining trucks and loaders
- **Carnegie Mellon Robotics Institute** — DARPA SubT Challenge competitors demonstrated autonomous navigation in subterranean environments
- **Micro-TBM technology** — Herrenknecht, Robbins produce small-diameter boring machines
- **The Boring Company** — demonstrated reduced-cost, smaller-scale tunnel boring

The ESCC excavation robot draws from all of these but requires novel integration of excavation, lining, and compact foldable form factor not present in any existing system.

## 4. Feasibility Assessment

### 4.1 Favorable Factors

- All surface equipment (drill rigs, concrete pumps, generators) is COTS or near-COTS, requiring only military hardening and packaging
- 1m shaft boring in soil/soft rock is well-understood commercial practice
- 30-50cm boreholes are standard well-drilling operations
- Shotcrete application in tunnels is a mature, daily-use technology
- Autonomous underground navigation was demonstrated extensively in DARPA SubT Challenge
- Electric-powered underground mining equipment is commercially available

### 4.2 Technical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Variable geology at deployment sites | High | Pre-survey likely theaters; rapid GPR assessment on arrival; modular cutting heads for soil vs. soft rock |
| Excavation robot foldable mechanism reliability | Medium | Proven deployment mechanisms from space robotics; extensive testing program |
| Water table interference | Medium | Dedicated dewatering shaft; grouting capability; site selection criteria |
| Chamber structural stability during excavation | Medium | Immediate shotcrete application; conservative chamber geometry; real-time ground monitoring sensors |
| Spoil removal through 1m shaft | Low | Bucket hoist or pneumatic conveyance; volumes are manageable (~100-200m³ for usable chamber) |

### 4.3 Geological Constraints

ESCC is optimized for **consolidated soil, clay, and soft sedimentary rock** — which covers the majority of terrain in European, Middle Eastern, and Pacific theater deployment zones. Hard crystalline rock (granite, basalt) would significantly slow operations and may require alternative approaches.

## 5. Operational Impact

### 5.1 Survivability Analysis

At 20-30m depth in earth:

- **Immune to**: all current UAS/drone threats, artillery, most guided munitions
- **Survivable against**: GBU-31 (JDAM), most cruise missiles
- **Requires dedicated bunker-buster**: GBU-28 (effective to ~30m earth) or GBU-57 MOP (~60m earth) — these are strategic-level weapons with limited inventory, high cost, and requiring large manned aircraft for delivery

Forcing the adversary to expend strategic-level munitions against an expedient facility that can be rebuilt in a week is an extremely favorable exchange ratio.

### 5.2 Signature Reduction

- Surface footprint: 5 shaft heads (easily camouflaged) + offset generator
- No vehicle concentration after construction (engineering vehicles depart)
- Antenna elements dispersed and connected via buried hardline
- Thermal signature minimal (underground is naturally temperature-stable)

### 5.3 Operational Flexibility

- Deploy forward with the advance party; bunker ready for main C2 element in 5-7 days
- Multiple bunkers possible — division HQ, brigade HQs, critical communications nodes
- Scalable chamber size — small relay post to full command center
- Reusable if position is retained; abandoned at low cost if situation requires displacement

## 6. Proposed Program Structure

### Phase 1: Concept Development (12 months, ~$2M)

- Detailed engineering analysis and simulation of excavation sequences
- Geological survey of representative deployment terrain types
- Excavation robot conceptual design and kinematic simulation
- Laboratory testing of foldable mechanism prototypes
- Construction timeline modeling under various soil conditions

### Phase 2: Prototype Development (18 months, ~$8M)

- Build and test excavation robot prototype (surface testing, then shaft deployment)
- Integrate with commercial drill rig and concrete pump
- Conduct full-sequence demonstration in controlled test site (known geology)
- Refine teleoperation interface and begin autonomous capability development

### Phase 3: Field Demonstration (12 months, ~$5M)

- Full system integration with military-representative vehicle package
- Demonstration at multiple sites with different geological conditions
- Operational timeline validation (target: 7 days to operational)
- Military user evaluation with Army Engineer and Signal units

**Total estimated program: 42 months, ~$15M**

## 7. Team and Qualifications

**Principal Investigator: [Your Name]**
- Senior Member, IEEE (since 1989 — 37 years)
- Lead Consultant, AT&T — networks, infrastructure, and federal systems
- Faculty, San Jose State University — Computer Engineering / Data Science
- Faculty, Sonoma State University
- Research interests: autonomous systems, cognitive architectures, HPC

**Organizational advantages:**
- AT&T Federal is a major DoD/IC contractor with existing relationships across defense agencies, cleared facilities, and program management infrastructure. AT&T's federal division provides networking, cybersecurity, and cloud services to DoD and could support ESCC communications integration, secure C2 networking within the bunker, and transition planning through established contract vehicles (e.g., Enterprise Infrastructure Solutions (EIS), FirstNet).
- SJSU and SSU provide access to graduate research students, laboratory facilities, and a pipeline for workforce development in autonomous systems engineering.

**Proposed collaborators (to be confirmed):**
- [University robotics lab — excavation robot design, autonomy, and DARPA SubT expertise]
- [Commercial boring/tunneling partner — drill rig military adaptation and field testing]
- [Military engineering SME — operational requirements, doctrine development, user evaluation]
- [AT&T Federal — secure communications architecture for subterranean C2 networking]

## 8. Conclusion

The drone threat has created an urgent operational need for expedient deep underground C2 facilities. ESCC addresses this need by integrating proven boring and concrete technologies with a novel foldable autonomous excavation robot, delivered as a 5-vehicle military engineering package capable of constructing a functional command bunker in 5-7 days.

The majority of the system leverages existing commercial technology at high TRL. The critical development item — the foldable excavation robot — builds on demonstrated capabilities from the mining automation and DARPA SubT communities. The proposed 42-month, $15M program provides a clear path from concept to field demonstration.

ESCC converts the adversary's drone advantage into a strategic-level targeting problem while providing persistent, survivable C2 at operationally relevant timescales and costs.

---

*Distribution Statement: [TBD — likely Distribution A: Approved for Public Release]*
