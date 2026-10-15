---
doc_id: "MANUAL-WELD-06"
title: "Northline Precision — WELD-06 Maintenance Manual"
asset: "WELD-06"
doc_type: "manual"
revision: "B"
effective_date: "2025-04-10"
supersedes: null
---

# Northline Precision — WELD-06 Maintenance Manual

**Asset:** WELD-06 — 6-axis robotic MIG welder (NL-RW6)  
**Cell:** Cell 4 · **Plant:** Building B — Cell 4 (precision machining & assembly)

## §1 Safety

- Verify fume extraction is running before striking an arc.
- Wear a welding helmet with a minimum shade 10 lens.

Reference: Northline EHS-STD-11 (Cell 4 lockout/tagout & PPE). Main disconnect: **480 V** at MCC-4.

## §2 Daily PM

- **Wire feed** — Confirm drive-roll tension per §4 and check for bird-nesting.
- **Torch** — Trim the contact tip and confirm gas flow at 18–22 CFH.
- **Fume arm** — Confirm extraction airflow at the hood.

## §3 Fault Codes

| Code | Meaning |
| --- | --- |
| E-815 | Weld wire-feed fault — wire slip exceeds 5 % of commanded speed. |
| E-604 | E-stop circuit open — safety relay dropped out. |
| E-512 | Encoder communication loss — drive feedback timeout above 50 ms. |

## §4 Torque Specs

| Fastener / setting | Value | Unit | Notes |
| --- | --- | --- | --- |
| Wire-feed drive-roll tension | 8 | N | Set with the tension knob scale. |
| Torch collar | 6 | N·m |  |
| Fixture clamps | 28 | N·m |  |

## §5 Consumables

- Contact tips: NL-CT-1.2 (every 8 h of arc)
- Liner: NL-WL-12 (every 500 h)
