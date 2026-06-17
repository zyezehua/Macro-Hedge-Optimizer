"""Implied-volatility surface indexed by moneyness (K/S) x maturity (years).

Built from a user-pasted Excel grid. Provides bilinear interpolation (flat extrapolation
at the edges) and per-scenario shock transforms (parallel shift / skew twist).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class VolSurface:
    """A rectangular IV grid.

    moneyness : 1-D array of K/S levels, ascending (e.g. 0.80, 0.90, 1.00, 1.10).
    maturities: 1-D array of times to expiry in years, ascending (e.g. 0.08, 0.25, 0.50).
    vols      : 2-D array shape (len(moneyness), len(maturities)) of implied vols (decimals).
    """

    moneyness: np.ndarray
    maturities: np.ndarray
    vols: np.ndarray

    def __post_init__(self) -> None:
        self.moneyness = np.asarray(self.moneyness, dtype=float)
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.vols = np.asarray(self.vols, dtype=float)
        if self.vols.shape != (len(self.moneyness), len(self.maturities)):
            raise ValueError(
                f"vols shape {self.vols.shape} != (moneyness {len(self.moneyness)}, "
                f"maturities {len(self.maturities)})"
            )
        if not (np.all(np.diff(self.moneyness) > 0) and np.all(np.diff(self.maturities) > 0)):
            raise ValueError("moneyness and maturities must be strictly ascending.")

    def iv(self, moneyness: float, T: float, warn_extrapolate: bool = False) -> float:
        """Bilinear interpolation in (moneyness, T) with flat (clamped) extrapolation."""
        m_lo, m_hi = self.moneyness[0], self.moneyness[-1]
        t_lo, t_hi = self.maturities[0], self.maturities[-1]
        if warn_extrapolate and (
            moneyness < m_lo or moneyness > m_hi or T < t_lo or T > t_hi
        ):
            warnings.warn(
                f"Extrapolating surface at moneyness={moneyness:.3f}, T={T:.3f} "
                f"(grid moneyness [{m_lo:.2f},{m_hi:.2f}], T [{t_lo:.2f},{t_hi:.2f}]).",
                stacklevel=2,
            )
        m = float(np.clip(moneyness, m_lo, m_hi))
        t = float(np.clip(T, t_lo, t_hi))

        i = int(np.clip(np.searchsorted(self.moneyness, m) - 1, 0, len(self.moneyness) - 2))
        j = int(np.clip(np.searchsorted(self.maturities, t) - 1, 0, len(self.maturities) - 2))

        m0, m1 = self.moneyness[i], self.moneyness[i + 1]
        t0, t1 = self.maturities[j], self.maturities[j + 1]
        wm = 0.0 if m1 == m0 else (m - m0) / (m1 - m0)
        wt = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)

        v00, v01 = self.vols[i, j], self.vols[i, j + 1]
        v10, v11 = self.vols[i + 1, j], self.vols[i + 1, j + 1]
        v0 = v00 * (1 - wt) + v01 * wt
        v1 = v10 * (1 - wt) + v11 * wt
        return float(max(v0 * (1 - wm) + v1 * wm, 1e-4))

    def iv_for_strike(self, S: float, K: float, T: float, warn_extrapolate: bool = False) -> float:
        return self.iv(K / S, T, warn_extrapolate=warn_extrapolate)

    # --- shock transforms -------------------------------------------------

    def shocked(self, vol_shock: float, mode: str = "parallel", twist_per_moneyness: float = 0.0) -> "VolSurface":
        """Return a new surface with a scenario vol shock applied.

        parallel  : add `vol_shock` (in vol decimals, e.g. 0.05 = +5 vol pts) to every node.
        skew_twist: parallel shift plus a rotation around ATM (moneyness=1.0); each node also
                    gets `twist_per_moneyness * (1.0 - moneyness)` so downside vols rise more.
        """
        new = self.vols + vol_shock
        if mode == "skew_twist" and twist_per_moneyness:
            tilt = twist_per_moneyness * (1.0 - self.moneyness)[:, None]
            new = new + tilt
        elif mode not in ("parallel", "skew_twist"):
            raise ValueError(f"Unknown vol-shock mode: {mode!r}")
        new = np.clip(new, 1e-4, None)
        return VolSurface(self.moneyness.copy(), self.maturities.copy(), new)
