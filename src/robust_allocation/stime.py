"""Stime di rendimenti attesi e covarianze, con lo shrinkage di Ledoit-Wolf verso la correlazione costante."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODI_ANNO = 12


@dataclass
class Stima:
    media: pd.Series  # rendimenti attesi annui
    covarianza: pd.DataFrame  # covarianza annua
    intensita: float  # peso del bersaglio nello shrinkage (0 = campionaria, 1 = bersaglio)


def covarianza_ledoit_wolf(r: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, float]:
    """Covarianza per periodo con shrinkage verso la matrice a correlazione costante.

    Ledoit e Wolf (2004), "Honey, I shrunk the sample covariance matrix": l'intensità ottimale minimizza
    l'errore quadratico atteso ed è stimata dai dati, senza parametri da scegliere.
    """
    x = np.asarray(r, dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    var = np.diag(s)
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    f = rbar * np.outer(sd, sd)
    np.fill_diagonal(f, var)

    y = x**2
    pi_mat = y.T @ y / t - s**2
    pi_hat = pi_mat.sum()
    theta = (x**3).T @ x / t - var[:, None] * s
    np.fill_diagonal(theta, 0.0)
    rho_hat = np.trace(pi_mat) + rbar * (np.outer(1 / sd, sd) * theta).sum()
    gamma = np.linalg.norm(f - s, "fro") ** 2
    kappa = (pi_hat - rho_hat) / gamma if gamma > 0 else 0.0
    intensita = float(np.clip(kappa / t, 0.0, 1.0))
    return intensita * f + (1 - intensita) * s, intensita


def stima(r: pd.DataFrame, shrinkage: bool = True) -> Stima:
    """Media storica e covarianza annualizzate."""
    if shrinkage:
        c, k = covarianza_ledoit_wolf(r)
    else:
        c, k = np.cov(r.to_numpy(), rowvar=False, ddof=0), 0.0
    cov = pd.DataFrame(c * PERIODI_ANNO, index=r.columns, columns=r.columns)
    return Stima(r.mean() * PERIODI_ANNO, cov, k)


def correlazione(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)
