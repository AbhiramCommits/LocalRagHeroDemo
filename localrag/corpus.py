"""Synthetic shop-floor corpus generator for the Northline Precision "Cell 4" demo.

Everything in this module is fictional, but it is *internally consistent*: fault
codes, part numbers, torque values, tolerances, inspection intervals and the
supersede chain all line up across maintenance manuals, spec sheets, QA work
instructions and incident notes. That consistency is what lets the agentic loop
be graded automatically.

`build_corpus(out_dir)` writes 40 Markdown documents:

  * 12 maintenance manuals   MANUAL-<ASSET>.md   (§1 Safety … §5 Consumables)
  * 10 machine spec sheets   SPEC-<ASSET>.md      (tolerances, torque, coolant PSI)
  * 10 QA work instructions  QC-WI-0NN.md         (pass/fail criteria + intervals)
  *  8 incident / changelog  INC-2026-0NN.md      (supersede older manual values)
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Single source of truth: fault codes.  E-214 means the same thing everywhere. #
# --------------------------------------------------------------------------- #
FAULT_CODES: dict[str, str] = {
    "E-101": "Coolant low-pressure trip — supply pressure below 20 PSI for more than 3 s.",
    "E-150": "Conveyor belt-tracking fault — belt edge offset exceeds 15 mm.",
    "E-214": "Spindle over-temperature — front spindle bearing above 78 °C.",
    "E-233": "Axis servo following error — position deviation exceeds 0.05 mm during rapid traverse.",
    "E-318": "Hydraulic pressure loss — system pressure below 1800 PSI.",
    "E-402": "Tool-changer jam — gripper did not confirm within 4 s.",
    "E-455": "Door-interlock fault — guard was opened while a cycle was active.",
    "E-512": "Encoder communication loss — drive feedback timeout above 50 ms.",
    "E-604": "E-stop circuit open — safety relay dropped out.",
    "E-707": "Laser beam-delivery fault — protective window contaminated.",
    "E-815": "Weld wire-feed fault — wire slip exceeds 5 % of commanded speed.",
    "E-921": "Chiller refrigerant low — chiller return temperature above setpoint + 4 °C.",
}

CELL = "Cell 4"
SITE = "Northline Precision"
PLANT = "Building B — Cell 4 (precision machining & assembly)"
SAFETY_STANDARD = "Northline EHS-STD-11 (Cell 4 lockout/tagout & PPE)"

# --------------------------------------------------------------------------- #
# The 12 Cell 4 assets                                                         #
# --------------------------------------------------------------------------- #
ASSETS: list[dict] = [
    {
        "id": "CNC-07",
        "name": "5-axis CNC machining centre",
        "model": "NL-VMC500 (HSK-A63)",
        "manual_rev": "C",
        "manual_date": "2025-06-15",
        "spec_rev": "B",
        "spec_date": "2025-04-02",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-214", "E-233", "E-402"],
        "safety": [
            "Lock out the 480 V main disconnect at MCC-4 before removing any way cover.",
            "Wear safety glasses and cut-resistant gloves when handling HSK toolholders.",
            "Never reach into the work envelope during the 200 RPM spin-down dwell.",
        ],
        "daily_pm": [
            ("Coolant", "Check concentration 6–8 % with a refractometer and skim tramp oil."),
            ("Way lube", "Confirm the reservoir is above MIN; top up with ISO VG-68."),
            ("Air supply", "Confirm 90–110 PSI at the FRL unit and drain the water trap."),
            ("Spindle warm-up", "Run the 10-minute warm-up cycle and listen for bearing noise."),
        ],
        "torque": [
            ("Spindle clamp bolts", "45", "N·m", "Cross-tighten in three passes."),
            ("Drawbar force", "12", "kN", "Measured with a drawbar gauge."),
            ("Way-cover M8 bolts", "22", "N·m", ""),
            ("Vise jaw bolts", "35", "N·m", ""),
        ],
        "consumables": [
            ("Coolant", "Blaser Vasco 5000, 6–8 %", "replenish weekly"),
            ("Way lube", "ISO VG-68", "1.2 L reservoir"),
            ("Spindle air filter", "NL-4471", "every 500 h"),
        ],
        "tolerances": [("X/Y/Z positioning", "±0.005", "mm"), ("Spindle runout (TIR)", "0.003", "mm")],
        "specs": [
            ("Spindle taper", "HSK-A63"),
            ("Max spindle speed", "15 000 RPM"),
            ("Rapid traverse", "40 m/min"),
            ("Controller", "Siemens 840D sl"),
            ("Connected load", "480 V / 3-phase / 40 A"),
        ],
    },
    {
        "id": "CNC-12",
        "name": "3-axis CNC turning centre",
        "model": "NL-T320 (12-station turret)",
        "manual_rev": "B",
        "manual_date": "2024-11-08",
        "spec_rev": "A",
        "spec_date": "2024-09-19",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-214", "E-233", "E-512"],
        "safety": [
            "Confirm the chuck is closed and the bar feeder isolated before opening the door.",
            "Keep clear of the turret index path when the machine is powered.",
            "Use the chip conveyor stop before clearing stringy chips.",
        ],
        "daily_pm": [
            ("Chuck pressure", "Confirm 350–420 PSI at the chuck gauge."),
            ("Turret", "Wipe the index couplings and apply a light film of NL-3310 grease."),
            ("Coolant", "Check concentration 6–8 % and confirm flow at the nozzle."),
            ("Chip conveyor", "Run 30 s and confirm the auger turns freely."),
        ],
        "torque": [
            ("Turret clamp bolts", "38", "N·m", "Even tightening, star pattern."),
            ("Chuck jaw bolts", "25", "N·m", "Use a soft-jaw torque wrench."),
            ("Bar-feeder collet nut", "18", "N·m", ""),
        ],
        "consumables": [
            ("Coolant", "Blaser Vasco 5000, 6–8 %", "replenish weekly"),
            ("Turret grease", "NL-3310", "every 250 h"),
            ("Chip conveyor chain", "ISO VG-220", "monthly"),
        ],
        "tolerances": [("X/Z positioning", "±0.006", "mm"), ("Chuck runout", "0.008", "mm")],
        "specs": [
            ("Bar capacity", "Ø 65 mm"),
            ("Max spindle speed", "4 500 RPM"),
            ("Turret stations", "12"),
            ("Controller", "Fanuc 0i-TF"),
            ("Connected load", "480 V / 3-phase / 30 A"),
        ],
    },
    {
        "id": "ROB-03",
        "name": "6-axis material-handling robot",
        "model": "NL-R20 (20 kg payload)",
        "manual_rev": "A",
        "manual_date": "2025-02-20",
        "spec_rev": "A",
        "spec_date": "2025-01-30",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-512", "E-604"],
        "safety": [
            "Enter the cell only with the robot in MANUAL mode and the teach pendant held.",
            "Respect the light curtain; never defeat the muting circuit.",
            "Support the arm before releasing any axis brake.",
        ],
        "daily_pm": [
            ("Gripper", "Confirm 72–80 PSI at the gripper regulator."),
            ("Axis brakes", "Cycle all six axes and listen for brake release."),
            ("Teach pendant", "Confirm the E-stop on the pendant drops the drives."),
        ],
        "torque": [
            ("Tool-plate bolts", "40", "N·m", "Use threadlocker NL-2201."),
            ("Base mounting M16", "180", "N·m", "Anchor grade 10.9."),
        ],
        "consumables": [
            ("Axis gear grease", "NL-RG2", "every 6 000 h"),
            ("Gripper pads", "NL-GP-88", "every 250 000 cycles"),
        ],
        "tolerances": [("Repeatability", "±0.05", "mm"), ("Path accuracy", "±0.15", "mm")],
        "specs": [
            ("Payload", "20 kg"),
            ("Reach", "1 810 mm"),
            ("Repeated pose accuracy", "±0.05 mm"),
            ("Controller", "NL-RC7"),
            ("Connected load", "480 V / 3-phase / 12 A"),
        ],
    },
    {
        "id": "PRS-02",
        "name": "250-ton hydraulic press",
        "model": "NL-HP250",
        "manual_rev": "D",
        "manual_date": "2025-09-12",
        "spec_rev": "B",
        "spec_date": "2025-07-25",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-318", "E-604", "E-101"],
        "safety": [
            "Install the die-block safety prop before any work inside the press bed.",
            "Bleed hydraulic pressure at the accumulator drain before opening a fitting.",
            "Keep hands clear of the die during the slow-close approach.",
        ],
        "daily_pm": [
            ("Hydraulic level", "Confirm the sight glass is between MIN and MAX."),
            ("Accumulator", "Check pre-charge at 1 450 PSI, ±30 PSI."),
            ("Die clamps", "Verify clamp bolts are snug before the first cycle."),
        ],
        "torque": [
            ("Die-clamp bolts", "120", "N·m", "Four corners, diagonal pattern."),
            ("Hydraulic relief pressure", "2500", "PSI", "Set with a calibrated gauge."),
        ],
        "consumables": [
            ("Hydraulic fluid", "ISO VG-46, 60 L", "condition monthly"),
            ("Filter element", "NL-HF-14", "every 1 000 h"),
        ],
        "tolerances": [("Bed parallelism", "0.05", "mm"), ("Ram repeatability", "±0.10", "mm")],
        "specs": [
            ("Rated tonnage", "250 ton"),
            ("Stroke", "400 mm"),
            ("Hydraulic pressure", "2 500 PSI nominal"),
            ("Controller", "NL-PC3"),
            ("Connected load", "480 V / 3-phase / 60 A"),
        ],
    },
    {
        "id": "CONV-05",
        "name": "modular belt conveyor",
        "model": "NL-CB12 (12 m)",
        "manual_rev": "B",
        "manual_date": "2024-12-04",
        "spec_rev": "A",
        "spec_date": "2024-10-11",
        "has_spec": True,
        "voltage": "240 V",
        "faults": ["E-150", "E-604"],
        "safety": [
            "Isolate the conveyor at the local disconnect before removing a belt section.",
            "Never track a running belt with a tool held in a gloved hand.",
        ],
        "daily_pm": [
            ("Belt tracking", "Confirm the belt stays within 5 mm of centre over one full lap."),
            ("Photo-eyes", "Wipe the lenses and confirm the transfer sensor triggers."),
        ],
        "torque": [
            ("Belt tension", "45", "N", "Measured with a tension gauge at the midpoint."),
            ("Drive-motor foot bolts", "24", "N·m", ""),
        ],
        "consumables": [
            ("Drive chain lube", "ISO VG-220", "monthly"),
            ("Belt splice pins", "NL-BS-12", "on wear"),
        ],
        "tolerances": [("Belt edge offset", "±5", "mm"), ("Belt speed", "0.35", "m/s")],
        "specs": [
            ("Length", "12 m"),
            ("Belt width", "400 mm"),
            ("Belt speed", "0.35 m/s"),
            ("Drive power", "240 V / 1-phase / 1.5 kW"),
        ],
    },
    {
        "id": "WASH-01",
        "name": "3-stage aqueous parts washer",
        "model": "NL-AW3",
        "manual_rev": "A",
        "manual_date": "2025-03-14",
        "spec_rev": "A",
        "spec_date": "2025-02-01",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-101", "E-921"],
        "safety": [
            "Allow the wash tank to cool below 55 °C before draining.",
            "Wear chemical splash goggles and nitrile gloves when handling detergent.",
        ],
        "daily_pm": [
            ("Wash tank", "Confirm the tank is at 60 °C ±3 and the level is above MIN."),
            ("Detergent", "Check concentration 2–4 % with a titration kit."),
            ("Rinse stage", "Confirm DI rinse conductivity below 30 µS/cm."),
        ],
        "torque": [
            ("Pump flange bolts", "18", "N·m", ""),
            ("Heater element", "12", "N·m", "Do not over-torque the ceramic seal."),
        ],
        "consumables": [
            ("Detergent", "NL-Clean 220, 2–4 %", "daily top-up"),
            ("Final rinse filter", "NL-WF-07", "weekly"),
        ],
        "tolerances": [("Wash temperature", "60", "°C ±3"), ("DI conductivity", "30", "µS/cm max")],
        "specs": [
            ("Stages", "wash / rinse / dry"),
            ("Tank volume", "180 L"),
            ("Wash temperature", "60 °C"),
            ("Connected load", "480 V / 3-phase / 24 A"),
        ],
    },
    {
        "id": "CMM-04",
        "name": "bridge coordinate-measuring machine",
        "model": "NL-BC900",
        "manual_rev": "C",
        "manual_date": "2025-08-01",
        "spec_rev": "B",
        "spec_date": "2025-06-16",
        "has_spec": True,
        "voltage": "240 V",
        "faults": ["E-512", "E-455"],
        "safety": [
            "Never move the gantry by hand while the air bearings are pressurised.",
            "Confirm the granite table is clean before a probe qualification.",
        ],
        "daily_pm": [
            ("Air supply", "Confirm 80–90 PSI at the air-bearing regulator."),
            ("Probe calibration", "Run the reference-sphere qualification; interval is 90 days."),
            ("Granite", "Wipe the table with a lint-free wipe and isopropyl alcohol."),
        ],
        "torque": [
            ("Probe module coupling", "2", "N·m", "Hand-tight plus a quarter turn."),
            ("Stylus tip", "0.5", "N·m", "Use the supplied torque driver."),
        ],
        "consumables": [
            ("Stylus tips", "NL-ST-2.0 (Ø2 mm ruby)", "on chipping"),
            ("Air filters", "NL-CF-05", "every 500 h"),
        ],
        "tolerances": [("Volumetric accuracy", "0.9", "µm"), ("Probe repeatability", "0.35", "µm")],
        "specs": [
            ("Measuring range", "900 × 700 × 600 mm"),
            ("Volumetric accuracy", "0.9 µm + L/350"),
            ("Probe", "Renishaw TP20"),
            ("Connected load", "240 V / 1-phase / 3 A"),
        ],
    },
    {
        "id": "LASER-02",
        "name": "3 kW fibre laser cutter",
        "model": "NL-FL3000",
        "manual_rev": "B",
        "manual_date": "2025-05-22",
        "spec_rev": "A",
        "spec_date": "2025-03-30",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-707", "E-921", "E-455"],
        "safety": [
            "Never operate with the enclosure interlocks bypassed; class-4 laser hazard.",
            "Wear laser-safety eyewear rated for 1 070 nm when the enclosure is open.",
        ],
        "daily_pm": [
            ("Protective window", "Inspect for spatter; clean with lens tissue if needed."),
            ("Chiller", "Confirm the return temperature is within setpoint + 2 °C."),
            ("Assist gas", "Confirm N2 supply; set pressure per the current revision (see §5)."),
        ],
        "torque": [
            ("Lens cassette screws", "3", "N·m", "Do not exceed — optics are fragile."),
            ("Focus-head bolts", "9", "N·m", ""),
        ],
        "consumables": [
            ("Protective windows", "NL-LW-30", "on contamination"),
            ("Assist gas (N2)", "99.999 %, pressure per §5", "continuous"),
        ],
        "tolerances": [("Positioning accuracy", "±0.05", "mm"), ("Kerf width", "0.20", "mm")],
        "specs": [
            ("Laser power", "3 kW"),
            ("Wavelength", "1 070 nm"),
            ("Cut area", "3 000 × 1 500 mm"),
            ("Connected load", "480 V / 3-phase / 32 A"),
        ],
    },
    {
        "id": "WELD-06",
        "name": "6-axis robotic MIG welder",
        "model": "NL-RW6",
        "manual_rev": "B",
        "manual_date": "2025-04-10",
        "spec_rev": "A",
        "spec_date": "2025-02-14",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-815", "E-604", "E-512"],
        "safety": [
            "Verify fume extraction is running before striking an arc.",
            "Wear a welding helmet with a minimum shade 10 lens.",
        ],
        "daily_pm": [
            ("Wire feed", "Confirm drive-roll tension per §4 and check for bird-nesting."),
            ("Torch", "Trim the contact tip and confirm gas flow at 18–22 CFH."),
            ("Fume arm", "Confirm extraction airflow at the hood."),
        ],
        "torque": [
            ("Wire-feed drive-roll tension", "8", "N", "Set with the tension knob scale."),
            ("Torch collar", "6", "N·m", ""),
            ("Fixture clamps", "28", "N·m", ""),
        ],
        "consumables": [
            ("Contact tips", "NL-CT-1.2", "every 8 h of arc"),
            ("Liner", "NL-WL-12", "every 500 h"),
        ],
        "tolerances": [("Leg length", "±0.5", "mm"), ("Weld position", "±0.4", "mm")],
        "specs": [
            ("Welding process", "MIG, pulsed"),
            ("Wire diameter", "1.2 mm"),
            ("Gas flow", "18–22 CFH"),
            ("Connected load", "480 V / 3-phase / 28 A"),
        ],
    },
    {
        "id": "GRND-03",
        "name": "CNC surface grinder",
        "model": "NL-SG500",
        "manual_rev": "B",
        "manual_date": "2024-10-16",
        "spec_rev": "A",
        "spec_date": "2024-08-28",
        "has_spec": True,
        "voltage": "480 V",
        "faults": ["E-214", "E-101", "E-402"],
        "safety": [
            "Stand clear of the wheel plane during spin-up; inspect the wheel for cracks.",
            "Wear a face shield in addition to safety glasses when dressing the wheel.",
        ],
        "daily_pm": [
            ("Wheel guard", "Confirm the guard is fastened and the wheel rated for the RPM."),
            ("Coolant", "Check concentration 4–6 % and confirm flow to the nozzle."),
            ("Wheel dressing", "Dress the wheel; interval is 40 h of grinding."),
        ],
        "torque": [
            ("Wheel arbor nut", "95", "N·m", "Left-hand thread; do not over-tighten."),
            ("Magnetic chuck bolts", "30", "N·m", ""),
        ],
        "consumables": [
            ("Grinding wheels", "NL-GW-250 (WA60)", "on wear"),
            ("Coolant", "NL-Grind 4–6 %", "weekly"),
        ],
        "tolerances": [("Surface finish (Ra)", "0.4", "µm"), ("Flatness", "0.005", "mm")],
        "specs": [
            ("Wheel diameter", "250 mm"),
            ("Max wheel speed", "2 850 RPM"),
            ("Table size", "500 × 200 mm"),
            ("Connected load", "480 V / 3-phase / 18 A"),
        ],
    },
    {
        "id": "DRLL-08",
        "name": "CNC drill / tap station",
        "model": "NL-DT400",
        "manual_rev": "A",
        "manual_date": "2025-01-09",
        "spec_rev": "A",
        "spec_date": "2024-12-01",
        "has_spec": False,
        "voltage": "480 V",
        "faults": ["E-233", "E-402", "E-214"],
        "safety": [
            "Confirm the spindle has stopped before changing a tap.",
            "Keep the work-holding clamps clear of the tool-change path.",
        ],
        "daily_pm": [
            ("Air supply", "Confirm 90–110 PSI at the FRL."),
            ("Tool changer", "Confirm each pocket indexes and the gripper confirms."),
        ],
        "torque": [
            ("Collet nut", "55", "N·m", "On the milling chuck."),
            ("Fixture clamp bolts", "30", "N·m", ""),
        ],
        "consumables": [
            ("Tapping fluid", "NL-Tap 22", "daily top-up"),
            ("Collets", "NL-CL-ER32", "on runout > 0.02 mm"),
        ],
        "tolerances": [("Hole position", "±0.03", "mm"), ("Hole diameter", "±0.02", "mm")],
        "specs": [
            ("Max drilling Ø", "20 mm"),
            ("Max tap", "M16"),
            ("Spindle speed", "8 000 RPM"),
            ("Connected load", "480 V / 3-phase / 22 A"),
        ],
    },
    {
        "id": "ASM-01",
        "name": "semi-automatic assembly & torque station",
        "model": "NL-AS1",
        "manual_rev": "A",
        "manual_date": "2025-07-03",
        "spec_rev": "A",
        "spec_date": "2025-05-20",
        "has_spec": False,
        "voltage": "240 V",
        "faults": ["E-455", "E-604", "E-815"],
        "safety": [
            "Confirm the two-hand control is functional before each shift.",
            "Keep the light curtain clear; the press platen is servo-driven.",
        ],
        "daily_pm": [
            ("Torque drivers", "Run the daily verification on the transducer (see QC-WI-007)."),
            ("Fixture", "Confirm locating pins are undamaged and seated."),
        ],
        "torque": [
            ("M8 safety-critical fasteners", "22", "N·m", "Angle-controlled; verify per QC-WI-007."),
            ("Fixture locking bolts", "15", "N·m", ""),
        ],
        "consumables": [
            ("Torque-driver bit sockets", "NL-BS-08", "on wear"),
            ("Locating pins", "NL-PN-08", "on wear"),
        ],
        "tolerances": [("Fastener torque", "±5", "%"), ("Position repeatability", "0.05", "mm")],
        "specs": [
            ("Drivers", "2 × servo torque driver"),
            ("Torque range", "5–50 N·m"),
            ("Connected load", "240 V / 1-phase / 6 A"),
        ],
    },
]

ASSET_BY_ID: dict[str, dict] = {a["id"]: a for a in ASSETS}

# --------------------------------------------------------------------------- #
# QA work instructions                                                        #
# --------------------------------------------------------------------------- #
WORK_INSTRUCTIONS: list[dict] = [
    {
        "wi": "QC-WI-001",
        "title": "Incoming raw-material verification (aluminium 6082-T6)",
        "rev": "B",
        "date": "2025-05-01",
        "asset": "ALL",
        "interval": "every incoming lot",
        "purpose": "Verify mill certificates and hardness of aluminium 6082-T6 billets before release to Cell 4.",
        "steps": [
            "Match the heat number on the billet to the mill certificate.",
            "Measure hardness in three locations with the Equotip; record the mean.",
            "Confirm the certificate is signed and traceable to EN 10204 3.1.",
        ],
        "pass": "Hardness 95–115 HB and a valid 3.1 certificate on file.",
        "fail": "Any reading outside 95–115 HB, or a certificate that does not match the heat number — quarantine the lot.",
        "tooling": "Equotip 550, calibrated calipers",
    },
    {
        "wi": "QC-WI-002",
        "title": "In-process first-article inspection",
        "rev": "C",
        "date": "2025-06-10",
        "asset": "CNC-07",
        "interval": "first part of every run + after any program change",
        "purpose": "Confirm the first part conforms before the run is released to production.",
        "steps": [
            "Measure all drawing dimensions marked critical-to-quality (CTQ).",
            "Record the measurements on the FAI sheet and initial it.",
            "Release the run only after QA signs the FAI sheet.",
        ],
        "pass": "Every CTQ dimension within its drawing tolerance.",
        "fail": "Any CTQ out of tolerance — stop the run, adjust the offset, re-cut.",
        "tooling": "CMM-04, bore gauges, thread gauges",
    },
    {
        "wi": "QC-WI-003",
        "title": "Surface-finish (Ra) inspection",
        "rev": "A",
        "date": "2025-02-12",
        "asset": "GRND-03",
        "interval": "every 20 parts",
        "purpose": "Verify the ground surface meets the drawing Ra requirement.",
        "steps": [
            "Clean the surface and take three Ra readings with the portable roughness tester.",
            "Compare the mean against the drawing callout.",
        ],
        "pass": "Mean Ra ≤ 0.4 µm and no single reading above 0.6 µm.",
        "fail": "Ra above 0.4 µm — dress the wheel (GRND-03 §2) and re-measure.",
        "tooling": "Mitutoyo SJ-210 roughness tester",
    },
    {
        "wi": "QC-WI-004",
        "title": "Thread pitch-diameter gauging",
        "rev": "B",
        "date": "2025-03-05",
        "asset": "CNC-12",
        "interval": "every 50 parts",
        "purpose": "Verify the pitch diameter of turned threads with calibrated gauges.",
        "steps": [
            "Select the go/no-go gauge matching the thread callout.",
            "The go gauge must enter by hand; the no-go must not enter more than two turns.",
        ],
        "pass": "Go enters fully, no-go does not exceed two turns.",
        "fail": "No-go enters more than two turns — rework or scrap the part.",
        "tooling": "Calibrated go/no-go thread gauges",
    },
    {
        "wi": "QC-WI-005",
        "title": "Hole-position tolerance check on the CMM",
        "rev": "D",
        "date": "2025-09-20",
        "asset": "CMM-04",
        "interval": "every 25 parts",
        "purpose": "Verify hole-position tolerance using CMM-04.",
        "steps": [
            "Qualify the probe on the reference sphere.",
            "Measure the datum features first, then the hole axes.",
            "Report the true-position deviation against the drawing.",
        ],
        "pass": "True-position deviation within 0.05 mm.",
        "fail": "Deviation above 0.05 mm — quarantine and raise a non-conformance.",
        "tooling": "CMM-04 with the Ø2 mm ruby stylus",
    },
    {
        "wi": "QC-WI-006",
        "title": "Weld bead visual and penetrant inspection",
        "rev": "B",
        "date": "2025-04-18",
        "asset": "WELD-06",
        "interval": "100 % visual, 10 % dye-penetrant",
        "purpose": "Verify weld quality on the Cell 4 structural bracket.",
        "steps": [
            "Inspect the bead visually for undercut, porosity and leg length.",
            "Apply dye penetrant to a 10 % sample and develop for 10 minutes.",
        ],
        "pass": "No cracks, leg length within ±0.5 mm, no penetrant indications.",
        "fail": "Any crack or linear indication — quarantine and re-weld.",
        "tooling": "Penetrant kit, weld gauges",
    },
    {
        "wi": "QC-WI-007",
        "title": "Torque verification of safety-critical fasteners",
        "rev": "C",
        "date": "2025-07-08",
        "asset": "ASM-01",
        "interval": "every shift (driver verification), 5 % audit per shift",
        "purpose": "Verify that safety-critical fasteners are torqued to specification.",
        "steps": [
            "Run the shift-start verification on the transducer to 22 N·m.",
            "Audit 5 % of finished assemblies by re-torquing to the spec.",
        ],
        "pass": "Driver within ±5 % of 22 N·m and no audited joint loosening.",
        "fail": "Driver out of tolerance — recalibrate before running production.",
        "tooling": "Calibrated torque transducer",
    },
    {
        "wi": "QC-WI-008",
        "title": "Laser cut edge-quality (dross) inspection",
        "rev": "A",
        "date": "2025-03-28",
        "asset": "LASER-02",
        "interval": "every 30 parts",
        "purpose": "Verify cut edge quality on laser-cut blanks.",
        "steps": [
            "Inspect the bottom edge for adherent dross.",
            "Measure the kerf width with the optical comparator.",
        ],
        "pass": "No adherent dross and kerf width 0.20 mm ±0.03 mm.",
        "fail": "Excessive dross or kerf out of range — adjust focus/gas and re-cut.",
        "tooling": "Optical comparator",
    },
    {
        "wi": "QC-WI-009",
        "title": "Coolant concentration and pH check",
        "rev": "B",
        "date": "2025-05-15",
        "asset": "CNC-07",
        "interval": "daily",
        "purpose": "Keep the CNC coolant within its controlled window.",
        "steps": [
            "Draw a sample and read concentration with the refractometer.",
            "Measure pH with a calibrated meter.",
        ],
        "pass": "Concentration 6–8 % and pH 8.5–9.5.",
        "fail": "Outside the window — adjust concentration or dump and recharge.",
        "tooling": "Refractometer, pH meter",
    },
    {
        "wi": "QC-WI-010",
        "title": "Final dimensional audit and packaging",
        "rev": "A",
        "date": "2025-08-22",
        "asset": "ALL",
        "interval": "100 % of shipped lots (sampled audit)",
        "purpose": "Final release gate before parts leave Cell 4.",
        "steps": [
            "Audit the lot against the CTQ list.",
            "Confirm the FAI and non-conformance records are complete.",
            "Apply the rust-preventive and package per the work order.",
        ],
        "pass": "Complete traceability records and conforming audit sample.",
        "fail": "Any missing record or non-conformance — hold the lot.",
        "tooling": "Calipers, CMM-04",
    },
]

# --------------------------------------------------------------------------- #
# Incident / changelog notes — these deliberately SUPERSEDE older values.     #
# --------------------------------------------------------------------------- #
INCIDENTS: list[dict] = [
    {
        "doc_id": "INC-2026-001",
        "title": "CNC-07 spindle clamp torque increase (fatigue failure RCA)",
        "rev": "A",
        "date": "2026-03-01",
        "supersedes": "MANUAL-CNC-07",
        "supersedes_section": "§4",
        "summary": (
            "RCA of two spindle-clamp bolt fatigue failures on CNC-07 concluded the "
            "manual torque of 45 N·m was insufficient for the HSK-A63 drawbar. Effective "
            "immediately the spindle clamp bolts must be torqued to 52 N·m."
        ),
        "changes": [
            ("Spindle clamp bolts", "45 N·m", "52 N·m"),
        ],
        "body": [
            "Fracture surfaces showed high-cycle fatigue originating at the thread root.",
            "Engineering analysis shows 52 N·m keeps preload above the drawbar separation load.",
            "MANUAL-CNC-07 §4 remains accurate for all other fasteners.",
        ],
    },
    {
        "doc_id": "INC-2026-002",
        "title": "PRS-02 hydraulic relief pressure re-set",
        "rev": "A",
        "date": "2026-02-18",
        "supersedes": "MANUAL-PRS-02",
        "supersedes_section": "§4",
        "summary": (
            "To eliminate intermittent E-318 trips during rapid approach, the hydraulic "
            "relief pressure on PRS-02 is raised from 2500 PSI to 2750 PSI."
        ),
        "changes": [("Hydraulic relief pressure", "2500 PSI", "2750 PSI")],
        "body": [
            "The higher setting stays below the accumulator safety limit of 3000 PSI.",
            "Update the machine placard after the change; record the gauge serial.",
        ],
    },
    {
        "doc_id": "INC-2026-003",
        "title": "QC-WI-005 hole-position tolerance tightened",
        "rev": "A",
        "date": "2026-01-29",
        "supersedes": "QC-WI-005",
        "supersedes_section": "§3",
        "summary": (
            "Customer PPAP feedback requires the hole-position true-position tolerance "
            "on the Cell 4 bracket to tighten from 0.05 mm to 0.03 mm."
        ),
        "changes": [("Hole-position tolerance", "0.05 mm", "0.03 mm")],
        "body": [
            "Applies to all Cell 4 brackets shipped after 2026-01-29.",
            "CMM-04 probe qualification interval is unchanged (see INC-2026-008).",
        ],
    },
    {
        "doc_id": "INC-2026-004",
        "title": "WELD-06 wire-feed drive-roll tension correction",
        "rev": "A",
        "date": "2026-02-05",
        "supersedes": "MANUAL-WELD-06",
        "supersedes_section": "§4",
        "summary": (
            "Excess E-815 events traced to drive-roll slip. Drive-roll tension is raised "
            "from 8 N to 11 N."
        ),
        "changes": [("Wire-feed drive-roll tension", "8 N", "11 N")],
        "body": [
            "Higher tension eliminated slip on 1.2 mm wire in a 30-part trial.",
            "Re-check the liner for kinks if slip persists at 11 N.",
        ],
    },
    {
        "doc_id": "INC-2026-005",
        "title": "LASER-02 assist-gas pressure increased for thick stock",
        "rev": "A",
        "date": "2026-03-11",
        "supersedes": "MANUAL-LASER-02",
        "supersedes_section": "§5",
        "summary": (
            "Nitrogen assist-gas pressure is raised from 180 PSI to 210 PSI to improve "
            "dross-free cutting of 6 mm mild steel."
        ),
        "changes": [("Assist-gas (N2) pressure", "180 PSI", "210 PSI")],
        "body": [
            "Lower pressures produced adherent dross above 5 mm thickness.",
            "Verify with QC-WI-008 edge inspection after the change.",
        ],
    },
    {
        "doc_id": "INC-2026-006",
        "title": "CONV-05 belt tension corrected after E-150 recurrence",
        "rev": "A",
        "date": "2026-01-15",
        "supersedes": "MANUAL-CONV-05",
        "supersedes_section": "§4",
        "summary": (
            "Recurrent E-150 belt-tracking faults traced to under-tension. Belt tension "
            "is corrected from 45 N to 58 N."
        ),
        "changes": [("Belt tension", "45 N", "58 N")],
        "body": [
            "The corrected tension holds the belt within 5 mm of centre over a full lap.",
            "Re-track the belt after any splice replacement.",
        ],
    },
    {
        "doc_id": "INC-2026-007",
        "title": "GRND-03 wheel-dressing interval shortened",
        "rev": "A",
        "date": "2026-02-27",
        "supersedes": "MANUAL-GRND-03",
        "supersedes_section": "§2",
        "summary": (
            "Surface-finish drift (Ra above 0.4 µm) between dresses requires shortening "
            "the wheel-dressing interval from 40 h to 25 h of grinding."
        ),
        "changes": [("Wheel-dressing interval", "40 h", "25 h")],
        "body": [
            "The shorter interval keeps Ra inside the QC-WI-003 window.",
            "Log each dress on the grinder PM sheet.",
        ],
    },
    {
        "doc_id": "INC-2026-008",
        "title": "CMM-04 probe calibration interval shortened",
        "rev": "A",
        "date": "2026-03-20",
        "supersedes": "MANUAL-CMM-04",
        "supersedes_section": "§2",
        "summary": (
            "Probe calibration interval is shortened from 90 days to 60 days after a "
            "drift event exceeded the 0.35 µm probe-repeatability budget."
        ),
        "changes": [("Probe calibration interval", "90 days", "60 days")],
        "body": [
            "CMM-04 must be re-qualified on the reference sphere every 60 days.",
            "Record the reference-sphere serial and the qualification result.",
        ],
    },
]


# Map each incident note to the asset it affects (used in front matter).
INCIDENT_ASSET: dict[str, str] = {
    "INC-2026-001": "CNC-07",
    "INC-2026-002": "PRS-02",
    "INC-2026-003": "CMM-04",
    "INC-2026-004": "WELD-06",
    "INC-2026-005": "LASER-02",
    "INC-2026-006": "CONV-05",
    "INC-2026-007": "GRND-03",
    "INC-2026-008": "CMM-04",
}


# --------------------------------------------------------------------------- #
# Rendering helpers                                                           #
# --------------------------------------------------------------------------- #
def _front_matter(**kw: object) -> str:
    lines = ["---"]
    for key, value in kw.items():
        if value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f'{key}: "{value}"' if isinstance(value, str) else f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def _fault_table(codes: list[str]) -> str:
    rows = ["| Code | Meaning |", "| --- | --- |"]
    for code in codes:
        rows.append(f"| {code} | {FAULT_CODES[code]} |")
    return "\n".join(rows)


def render_manual(asset: dict) -> str:
    a = asset
    fm = _front_matter(
        doc_id=f"MANUAL-{a['id']}",
        title=f"{SITE} — {a['id']} Maintenance Manual",
        asset=a["id"],
        doc_type="manual",
        revision=a["manual_rev"],
        effective_date=a["manual_date"],
        supersedes=None,
    )
    safety = "\n".join(f"- {s}" for s in a["safety"])
    pm = "\n".join(f"- **{k}** — {v}" for k, v in a["daily_pm"])
    torque_rows = ["| Fastener / setting | Value | Unit | Notes |", "| --- | --- | --- | --- |"]
    for item, val, unit, note in a["torque"]:
        torque_rows.append(f"| {item} | {val} | {unit} | {note} |")
    cons = "\n".join(f"- {k}: {v} ({i})" for k, v, i in a["consumables"])
    return f"""{fm}

# {SITE} — {a['id']} Maintenance Manual

**Asset:** {a['id']} — {a['name']} ({a['model']})  
**Cell:** {CELL} · **Plant:** {PLANT}

## §1 Safety

{safety}

Reference: {SAFETY_STANDARD}. Main disconnect: **{a['voltage']}** at MCC-4.

## §2 Daily PM

{pm}

## §3 Fault Codes

{_fault_table(a['faults'])}

## §4 Torque Specs

{chr(10).join(torque_rows)}

## §5 Consumables

{cons}
"""


def render_spec(asset: dict) -> str:
    a = asset
    fm = _front_matter(
        doc_id=f"SPEC-{a['id']}",
        title=f"{SITE} — {a['id']} Machine Specification Sheet",
        asset=a["id"],
        doc_type="spec",
        revision=a["spec_rev"],
        effective_date=a["spec_date"],
        supersedes=None,
    )
    specs = "\n".join(f"| {k} | {v} |" for k, v in a["specs"])
    tols = "\n".join(f"| {k} | {v} | {u} |" for k, v, u in a["tolerances"])
    torque_rows = "\n".join(
        f"| {item} | {val} | {unit} |" for item, val, unit, _ in a["torque"]
    )
    return f"""{fm}

# {SITE} — {a['id']} Machine Specification Sheet

**Model:** {a['model']} · **Cell:** {CELL}

## §1 General Specifications

| Parameter | Value |
| --- | --- |
{specs}

## §2 Tolerances

| Feature | Tolerance | Unit |
| --- | --- | --- |
{tols}

## §3 Torque and Pressure Settings

| Setting | Value | Unit |
| --- | --- | --- |
{torque_rows}

## §4 Utilities

- Electrical: {a['voltage']} at MCC-4
- Air: 90–110 PSI clean, dry
"""


def render_wi(wi: dict) -> str:
    fm = _front_matter(
        doc_id=wi["wi"],
        title=f"{SITE} — {wi['title']}",
        asset=wi["asset"],
        doc_type="work_instruction",
        revision=wi["rev"],
        effective_date=wi["date"],
        supersedes=None,
    )
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(wi["steps"]))
    return f"""{fm}

# {SITE} — QA Work Instruction {wi['wi']}

**Title:** {wi['title']}  
**Asset:** {wi['asset']} · **Inspection interval:** {wi['interval']}

## §1 Purpose

{wi['purpose']}

## §2 Procedure

{steps}

## §3 Pass / Fail Criteria

- **PASS:** {wi['pass']}
- **FAIL:** {wi['fail']}

## §4 Tooling

{wi['tooling']}
"""


def render_incident(inc: dict) -> str:
    fm = _front_matter(
        doc_id=inc["doc_id"],
        title=f"{SITE} — {inc['title']}",
        asset=INCIDENT_ASSET.get(inc["doc_id"], inc["supersedes"]),
        doc_type="incident",
        revision=inc["rev"],
        effective_date=inc["date"],
        supersedes=inc["supersedes"],
        supersedes_section=inc["supersedes_section"],
    )
    changes = "\n".join(f"| {k} | {old} | {new} |" for k, old, new in inc["changes"])
    body = "\n".join(f"- {b}" for b in inc["body"])
    return f"""{fm}

# {SITE} — Incident / Change Note {inc['doc_id']}

**Supersedes:** {inc['supersedes']} {inc['supersedes_section']} · **Effective:** {inc['date']}

## §1 Summary

{inc['summary']}

## §2 Change Record

| Item | Previous value | New value |
| --- | --- | --- |
{changes}

## §3 Engineering Notes

{body}
"""


def build_corpus(out_dir: str | Path = "data/corpus") -> list[Path]:
    """Write the full synthetic corpus and return the list of files created."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    for asset in ASSETS:
        _write(f"MANUAL-{asset['id']}.md", render_manual(asset))
    for asset in ASSETS:
        if asset["has_spec"]:
            _write(f"SPEC-{asset['id']}.md", render_spec(asset))
    for wi in WORK_INSTRUCTIONS:
        _write(f"{wi['wi']}.md", render_wi(wi))
    for inc in INCIDENTS:
        _write(f"{inc['doc_id']}.md", render_incident(inc))

    return written


if __name__ == "__main__":  # pragma: no cover - manual helper
    files = build_corpus()
    print(f"wrote {len(files)} documents")
