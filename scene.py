"""
Scene dataclasses and JSON I/O for coils and implants.
Includes ChipProfile metadata for Starfish mode (no physics coupling).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List
import numpy as np


@dataclass
class ChipProfile:
    name: str = "starfish_placeholder"
    carrier_hz: float = 3.4e5
    me_sensitivity: float = 1.0
    rectifier_eff: float = 0.7
    power_budget_mw: float = 1.0


@dataclass
class Coil:
    pos: List[float]
    moment: List[float]
    active: bool = True


@dataclass
class Implant:
    pos: List[float]
    axis: List[float]


@dataclass
class Scene:
    coils: List[Coil]
    implants: List[Implant]
    name: str = "scene"
    chip: ChipProfile = ChipProfile()
    layout_note: str = ""

    @staticmethod
    def from_json(path: str) -> "Scene":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        coils = [Coil(**c) for c in data["coils"]]
        implants = [Implant(**i) for i in data["implants"]]
        chip = ChipProfile(**data.get("chip", {}))
        return Scene(coils=coils, implants=implants, name=data.get("name", "scene"), chip=chip, layout_note=data.get("layout_note", ""))

    def to_json(self, path: str) -> None:
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def scene_arrays(scene: Scene):
    """Return np arrays for coil positions/moments and implant positions/axes."""
    coil_pos = np.array([c.pos for c in scene.coils], dtype=float)
    coil_mom = np.array([c.moment for c in scene.coils], dtype=float)
    imp_pos = np.array([i.pos for i in scene.implants], dtype=float)
    imp_axis = np.array([i.axis for i in scene.implants], dtype=float)
    return coil_pos, coil_mom, imp_pos, imp_axis


