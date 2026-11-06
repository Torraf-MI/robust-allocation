"""Metriche di rendimento, rischio e diversificazione."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cvar import cvar_empirico
from .stime import PERIODI_ANNO


def rendimento_annuo(r: pd.Series) -> float:
    anni = len(r) / PERIODI_ANNO
    return float((1 + r).prod() ** (1 / anni) - 1) if anni > 0 else np.nan


def volatilita(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(PERIODI_ANNO))


def drawdown(r: pd.Series) -> pd.Series:
    valore = (1 + r).cumprod()
    return valore / valore.cummax().clip(lower=1.0) - 1


def numero_effettivo(w: pd.Series | np.ndarray) -> float:
    """Numero di asset «equivalenti» (inverso dell'indice di Herfindahl): N per pesi uguali, 1 se concentrato."""
    w = np.asarray(w, dtype=float)
    return float(1 / np.sum(w**2))


def rapporto_diversificazione(w: pd.Series, cov: pd.DataFrame) -> float:
    """Media ponderata delle volatilità diviso volatilità del portafoglio: 1 senza diversificazione."""
    x = w.reindex(cov.index).fillna(0).to_numpy()
    s = cov.to_numpy()
    return float(x @ np.sqrt(np.diag(s)) / np.sqrt(x @ s @ x))


def contributi_rischio(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Quota di ogni asset nella varianza del portafoglio (somma 1)."""
    x = w.reindex(cov.index).fillna(0).to_numpy()
    s = cov.to_numpy()
    c = x * (s @ x)
    return pd.Series(c / c.sum(), index=cov.index)


def tabella(r: pd.Series, risk_free: float = 0.0, alfa: float = 0.95) -> dict[str, float]:
    ra, vol = rendimento_annuo(r), volatilita(r)
    dd = drawdown(r).min()
    return {
        "Rendimento annuo": ra,
        "Volatilità": vol,
        "Sharpe": (ra - risk_free) / vol if vol > 0 else np.nan,
        "Massima perdita": dd,
        "Calmar": ra / -dd if dd < 0 else np.nan,
        f"CVaR {alfa:.0%} mensile": cvar_empirico(r.to_numpy(), alfa),
        "Mese peggiore": float(r.min()),
    }
