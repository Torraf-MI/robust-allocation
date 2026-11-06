"""Frontiera ricampionata di Michaud: la media-varianza resa robusta agli errori di stima.

Le stime di media e covarianza sono incerte. Si simulano molte storie alternative compatibili con le stime
(stessa lunghezza del campione), si ottimizza su ciascuna e si fa la media dei pesi. Il portafoglio che ne esce è
più diversificato e cambia meno da una stima all'altra; la dispersione dei pesi tra le simulazioni misura quanto
è fragile l'ottimizzazione classica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ottimizzatore import Ottimizzatore, Vincoli
from .stime import PERIODI_ANNO


@dataclass
class RisultatoRicampionamento:
    pesi: pd.Series  # media dei pesi ottimi
    simulazioni: pd.DataFrame  # pesi ottimi di ogni simulazione (righe)

    @property
    def dispersione(self) -> pd.Series:
        return self.simulazioni.std().rename("Deviazione standard dei pesi")


def ricampiona(mu: pd.Series, cov: pd.DataFrame, osservazioni: int, avversione: float,
               vincoli: Vincoli | None = None, simulazioni: int = 200, seed: int = 0) -> RisultatoRicampionamento:
    """Media dei pesi media-varianza su ``simulazioni`` campioni di ``osservazioni`` mesi tratti da N(μ, Σ)."""
    rng = np.random.default_rng(seed)
    asset = list(cov.index)
    m = mu.reindex(asset).to_numpy() / PERIODI_ANNO
    s = cov.to_numpy() / PERIODI_ANNO
    chol = np.linalg.cholesky(s + 1e-12 * np.eye(len(asset)))
    base = vincoli or Vincoli()
    v = Vincoli(base.minimo, base.massimo, base.gruppi)  # i costi non c'entrano con l'errore di stima
    righe = []
    for _ in range(simulazioni):
        x = m + rng.standard_normal((osservazioni, len(asset))) @ chol.T
        mu_s = pd.Series(x.mean(axis=0) * PERIODI_ANNO, index=asset)
        cov_s = pd.DataFrame(np.cov(x, rowvar=False) * PERIODI_ANNO, index=asset, columns=asset)
        try:
            righe.append(Ottimizzatore(mu_s, cov_s, v).media_varianza(avversione).pesi)
        except RuntimeError:
            continue
    sim = pd.DataFrame(righe).reset_index(drop=True)
    pesi = sim.mean()
    return RisultatoRicampionamento((pesi / pesi.sum()).rename("Ricampionato"), sim)
