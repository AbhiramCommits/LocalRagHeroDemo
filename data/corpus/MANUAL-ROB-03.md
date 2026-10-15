---
doc_id: "MANUAL-ROB-03"
title: "Northline Precision — ROB-03 Maintenance Manual"
asset: "ROB-03"
doc_type: "manual"
revision: "A"
effective_date: "2025-02-20"
supersedes: null
---

# Northline Precision — ROB-03 Maintenance Manual

**Asset:** ROB-03 — 6-axis material-handling robot (NL-R20 (20 kg payload))  
**Cell:** Cell 4 · **Plant:** Building B — Cell 4 (precision machining & assembly)

## §1 Safety

- Enter the cell only with the robot in MANUAL mode and the teach pendant held.
- Respect the light curtain; never defeat the muting circuit.
- Support the arm before releasing any axis brake.

Reference: Northline EHS-STD-11 (Cell 4 lockout/tagout & PPE). Main disconnect: **480 V** at MCC-4.

## §2 Daily PM

- **Gripper** — Confirm 72–80 PSI at the gripper regulator.
- **Axis brakes** — Cycle all six axes and listen for brake release.
- **Teach pendant** — Confirm the E-stop on the pendant drops the drives.

## §3 Fault Codes

| Code | Meaning |
| --- | --- |
| E-512 | Encoder communication loss — drive feedback timeout above 50 ms. |
| E-604 | E-stop circuit open — safety relay dropped out. |

## §4 Torque Specs

| Fastener / setting | Value | Unit | Notes |
| --- | --- | --- | --- |
| Tool-plate bolts | 40 | N·m | Use threadlocker NL-2201. |
| Base mounting M16 | 180 | N·m | Anchor grade 10.9. |

## §5 Consumables

- Axis gear grease: NL-RG2 (every 6 000 h)
- Gripper pads: NL-GP-88 (every 250 000 cycles)
