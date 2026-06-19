"""Serialize / deserialize custom stress templates (JSON).

The app is stateless and runs on a shared public host, so user-defined stresses are not written
server-side. Instead a user downloads their custom library as JSON and re-uploads it later — the
data stays with the user, consistent with the in-session / nothing-stored privacy stance.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .library import StressTemplate
from .macro import InstrumentShock

_VOL_MODES = {"parallel", "skew_twist"}


def template_to_dict(t: StressTemplate) -> dict:
    return {"key": t.key, "name": t.name, "note": t.note,
            "equity": asdict(t.equity), "credit": asdict(t.credit)}


def _shock_from_dict(d: dict) -> InstrumentShock:
    sh = InstrumentShock(
        spot_shock=float(d["spot_shock"]),
        vol_shock=float(d.get("vol_shock", 0.0)),
        vol_mode=str(d.get("vol_mode", "parallel")),
        twist=float(d.get("twist", 0.0)),
    )
    if sh.vol_mode not in _VOL_MODES:
        raise ValueError(f"vol_mode must be one of {sorted(_VOL_MODES)}, got {sh.vol_mode!r}")
    return sh


def template_from_dict(d: dict) -> StressTemplate:
    try:
        return StressTemplate(
            key=str(d["key"]), name=str(d["name"]),
            equity=_shock_from_dict(d["equity"]), credit=_shock_from_dict(d["credit"]),
            note=str(d.get("note", "")),
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"Malformed stress template: missing/invalid field ({e}).") from e


def dump_templates(templates: dict[str, StressTemplate]) -> str:
    """JSON for a {key: StressTemplate} library (a list of template objects)."""
    return json.dumps([template_to_dict(t) for t in templates.values()], indent=2)


def load_templates(text: str) -> dict[str, StressTemplate]:
    """Parse a JSON library into {key: StressTemplate}. Raises ValueError on malformed input."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of stress templates.")
    out: dict[str, StressTemplate] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each stress template must be a JSON object.")
        t = template_from_dict(item)
        out[t.key] = t
    return out
