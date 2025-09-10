omniwpt — Omnidirectional Wireless Power Transfer Simulator

Minimal, clear, and faithful to the Active Echo (AE) concept: multi-coil magnetic field steering maintains power to tiny, randomly oriented implants. Pure NumPy + Matplotlib. Single-file CLI. No bloat.

Quick start
- python -m pip install -r requirements.txt
- python app.py --scene scenes/two_coils.json --axis 60 --depth 0.03

Outputs
- heatmap_single.png, heatmap_steered.png: 12×12 cm window at depth
- rotation.png: single vs steered vs angle (0–180°)
- summary.json: key metrics (min/median PTE, coverage, gain vs baseline)

Starfish mode
- Preset scenes under scenes/starfish_*.json
- Motion script with AE cadence (5–10 Hz): motion_trace.png
- Fairness steering (multi-implant worst-case boost): --fair

Files
- core.py: dipole B-field, PTE proxy, grid
- control.py: AE emulate, steering rules (single and fairness)
- scene.py: dataclasses (Coil, Implant, Scene, ChipProfile), JSON I/O
- viz.py: heatmaps and rotation sweep plots
- app.py: CLI, motion mode, metrics and summary.json
- tests.py: small acceptance tests via assert

Physics (10 lines)
- Use μ0 = 4π×1e−7.
- Dipole field at r from dipole moment m at r0: B = μ0/(4πr^3) [3 n (n·m) − m], n = (r−r0)/|r−r0|.
- Implant axis u from angle θ (deg) in x–z plane.
- PTE proxy: |B·u| (scaled to relative volts by ChipProfile.me_sensitivity; no physics change).
- AE emulate per coil k at point p: c_k = B_k(p)·u.
- Steering (single implant): w_k = sign(c_k)|c_k| / Σ|c_k|, scaled to budget.
- Fairness (multi-implant): maximize min_i |Σ_k w_k c_k^(i)| with an L1 budget (heuristic).

Constraints
- Only numpy and matplotlib. No other deps.
- Pure, vectorized functions; short files.
- No frameworks. Simple prints for CLI status.

Glossary
- B: magnetic flux density vector
- u: implant axis unit vector
- AE: Active Echo, using implant-side tiny coil to sense coupling and steer TX
- phase: binary (+/−) current polarity per coil


