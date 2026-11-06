"""Hierarchical Risk Parity (López de Prado, 2016).

Tre passi: raggruppa gli asset per somiglianza (clustering gerarchico sulla distanza di correlazione), li riordina
in modo che gli asset simili stiano vicini (quasi-diagonalizzazione) e divide il capitale con una bisezione
ricorsiva, in proporzione inversa alla varianza di ciascun gruppo. Non inverte la matrice di covarianza e non usa
i rendimenti attesi, quindi è molto meno sensibile agli errori di stima della media-varianza.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .stime import correlazione


@dataclass
class RisultatoHRP:
    pesi: pd.Series
    ordine: list[str]  # asset nell'ordine quasi-diagonale
    collegamenti: np.ndarray  # matrice di linkage di scipy, per il dendrogramma
    distanze: pd.DataFrame


def distanza_correlazione(corr: pd.DataFrame) -> pd.DataFrame:
    """d = √(½(1 − ρ)): 0 per asset identici, 1 per asset perfettamente opposti."""
    return np.sqrt(np.clip(0.5 * (1 - corr), 0.0, None))


def _varianza_cluster(cov: pd.DataFrame, membri: list[str]) -> float:
    c = cov.loc[membri, membri].to_numpy()
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def bisezione_ricorsiva(cov: pd.DataFrame, ordine: list[str]) -> pd.Series:
    pesi = pd.Series(1.0, index=ordine)
    gruppi = [ordine]
    while gruppi:
        nuovi = []
        for g in gruppi:
            if len(g) < 2:
                continue
            meta = len(g) // 2
            sx, dx = g[:meta], g[meta:]
            v_sx, v_dx = _varianza_cluster(cov, sx), _varianza_cluster(cov, dx)
            alfa = 1 - v_sx / (v_sx + v_dx)
            pesi[sx] *= alfa
            pesi[dx] *= 1 - alfa
            nuovi += [sx, dx]
        gruppi = nuovi
    return pesi


def hrp(cov: pd.DataFrame, metodo: str = "single") -> RisultatoHRP:
    """Pesi HRP dalla covarianza. ``metodo`` è il criterio di collegamento del clustering."""
    corr = correlazione(cov)
    dist = distanza_correlazione(corr)
    d = dist.to_numpy().copy()
    np.fill_diagonal(d, 0.0)
    link = linkage(squareform(d, checks=False), method=metodo)
    ordine = [cov.index[i] for i in leaves_list(link)]
    pesi = bisezione_ricorsiva(cov, ordine).reindex(cov.index).rename("HRP")
    return RisultatoHRP(pesi, ordine, link, dist)
