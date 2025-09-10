What this is
An open-source simulator that shows how multi-coil magnetic field steering + Active Echo keeps tiny battery-free implants powered across rotation and motion. It uses a fairness controller to guarantee worst-case nodes stay above threshold under a fixed power/SAR budget. Plug-and-play with your chip once specs are known.

Why you’ll care
- De-risks multi-region powering before first silicon.
- Lets you show labs a live animation of robustness vs orientation/motion.
- Outputs partner-friendly metrics (coverage, min-PTE, recovery after motion).
- Minimal code; easy to integrate your AE RX & coil drivers later.

What to change when you share specs
- ChipProfile: carrier_hz, me_sensitivity, rectifier_eff.
- Target implant depths and axis distributions.
- AE cadence and valid weight update rates.
- Current/SAR budget constraints.
- Coil geometry if you have a preferred array.

Next experiments
- Three-coil vs two-coil: quantify robustness and budget usage.
- Reacquisition latency under scripted head motion.
- Multi-lab preset: drop in actual room geometry and see coverage.


