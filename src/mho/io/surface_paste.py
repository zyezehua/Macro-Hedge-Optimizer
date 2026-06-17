"""Parse an Excel-pasted implied-vol grid into a VolSurface.

Expected layout (tab- or comma-separated, as copied from Excel):

    (label)   1M     3M     6M     1Y
    80%       32%    30%    28%    26%
    90%       26%    25%    24%    23%
    100%      20%    20%    20%    20%
    110%      18%    19%    20%    21%

Rows are moneyness (K/S), columns are maturities. The top-left cell is an optional label.
Moneyness, maturities and IVs accept several notations (see helpers below).
"""

from __future__ import annotations

import re

import numpy as np

from ..pricing.surface import VolSurface

_TENOR_UNITS = {"d": 1 / 365.0, "w": 7 / 365.0, "m": 1 / 12.0, "y": 1.0}


def parse_tenor(token: str) -> float:
    """'1M' -> 0.0833y, '2W' -> 0.038y, '0.5' / '0.5Y' -> 0.5y, '30D' -> 0.082y."""
    s = token.strip().lower()
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([dwmy]?)\s*", s)
    if not m:
        raise ValueError(f"Cannot parse maturity token: {token!r}")
    val = float(m.group(1))
    unit = m.group(2)
    return val * _TENOR_UNITS[unit] if unit else val  # bare number = years


def parse_moneyness(token: str) -> float:
    """'80%' -> 0.80, '0.8' -> 0.8, '80' -> 0.80 (>2 treated as percent)."""
    s = token.strip().replace("%", "")
    val = float(s)
    if "%" in token or val > 2.0:
        return val / 100.0
    return val


def parse_iv(token: str) -> float:
    """'20%' -> 0.20, '0.20' -> 0.20, '20' -> 0.20 (>1.5 treated as percent)."""
    s = token.strip().replace("%", "")
    val = float(s)
    if "%" in token or val > 1.5:
        return val / 100.0
    return val


def _split(line: str) -> list[str]:
    if "\t" in line:
        parts = line.split("\t")
    elif "," in line:
        parts = line.split(",")
    else:
        parts = line.split()
    return [p for p in (x.strip() for x in parts) if p != ""]


def parse_surface(text: str) -> VolSurface:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Need a header row of maturities plus at least one moneyness row.")

    header = _split(lines[0])
    # The header's first token is the (optional) corner label; the rest are maturities.
    maturity_tokens = header[1:] if len(header) > 1 else header
    maturities = [parse_tenor(t) for t in maturity_tokens]

    moneyness: list[float] = []
    rows: list[list[float]] = []
    for ln in lines[1:]:
        cells = _split(ln)
        if len(cells) != len(maturities) + 1:
            raise ValueError(
                f"Row {ln!r} has {len(cells)} cells; expected {len(maturities) + 1} "
                f"(1 moneyness + {len(maturities)} maturities)."
            )
        moneyness.append(parse_moneyness(cells[0]))
        rows.append([parse_iv(c) for c in cells[1:]])

    vols = np.array(rows, dtype=float)

    # Sort to strictly ascending axes (VolSurface requires it).
    m_order = np.argsort(moneyness)
    t_order = np.argsort(maturities)
    moneyness = np.array(moneyness)[m_order]
    maturities = np.array(maturities)[t_order]
    vols = vols[np.ix_(m_order, t_order)]
    return VolSurface(moneyness, maturities, vols)
