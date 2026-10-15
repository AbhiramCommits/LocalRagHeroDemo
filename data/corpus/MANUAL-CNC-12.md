---
doc_id: "MANUAL-CNC-12"
title: "Northline Precision — CNC-12 Maintenance Manual"
asset: "CNC-12"
doc_type: "manual"
revision: "B"
effective_date: "2024-11-08"
supersedes: null
---

# Northline Precision — CNC-12 Maintenance Manual

**Asset:** CNC-12 — 3-axis CNC turning centre (NL-T320 (12-station turret))  
**Cell:** Cell 4 · **Plant:** Building B — Cell 4 (precision machining & assembly)

## §1 Safety

- Confirm the chuck is closed and the bar feeder isolated before opening the door.
- Keep clear of the turret index path when the machine is powered.
- Use the chip conveyor stop before clearing stringy chips.

Reference: Northline EHS-STD-11 (Cell 4 lockout/tagout & PPE). Main disconnect: **480 V** at MCC-4.

## §2 Daily PM

- **Chuck pressure** — Confirm 350–420 PSI at the chuck gauge.
- **Turret** — Wipe the index couplings and apply a light film of NL-3310 grease.
- **Coolant** — Check concentration 6–8 % and confirm flow at the nozzle.
- **Chip conveyor** — Run 30 s and confirm the auger turns freely.

## §3 Fault Codes

| Code | Meaning |
| --- | --- |
| E-214 | Spindle over-temperature — front spindle bearing above 78 °C. |
| E-233 | Axis servo following error — position deviation exceeds 0.05 mm during rapid traverse. |
| E-512 | Encoder communication loss — drive feedback timeout above 50 ms. |

## §4 Torque Specs

| Fastener / setting | Value | Unit | Notes |
| --- | --- | --- | --- |
| Turret clamp bolts | 38 | N·m | Even tightening, star pattern. |
| Chuck jaw bolts | 25 | N·m | Use a soft-jaw torque wrench. |
| Bar-feeder collet nut | 18 | N·m |  |

## §5 Consumables

- Coolant: Blaser Vasco 5000, 6–8 % (replenish weekly)
- Turret grease: NL-3310 (every 250 h)
- Chip conveyor chain: ISO VG-220 (monthly)
