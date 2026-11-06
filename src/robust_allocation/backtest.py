"""Backtest walk-forward: a ogni ribilanciamento ogni strategia vede solo i mesi già trascorsi.

Alla fine del mese t − 1 la strategia stima i parametri sui rendimenti fino a t − 1 compreso e sceglie i pesi; il
rendimento del mese t è quello dei pesi scelti, meno i costi delle operazioni. Tra un ribilanciamento e l'altro i
pesi derivano con i prezzi, come in un portafoglio vero.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import black_litterman as bl
from .cvar import minimo_cvar
from .hrp import hrp
from .ottimizzatore import Ottimizzatore, Vincoli
from .ricampionamento import ricampiona
from .stime import stima

Strategia = Callable[[pd.DataFrame, pd.Series], pd.Series]


@dataclass
class RisultatoBacktest:
    nome: str
    rendimenti: pd.Series  # mensili, al netto dei costi
    pesi: pd.DataFrame  # pesi scelti a ogni ribilanciamento
    rotazione: pd.Series  # rotazione (metà della somma delle variazioni) a ogni ribilanciamento
    costi: pd.Series

    @property
    def rotazione_annua(self) -> float:
        anni = len(self.rendimenti) / 12
        return float(self.rotazione.sum() / anni) if anni > 0 else np.nan


def esegui(r: pd.DataFrame, strategia: Strategia, nome: str, minimo_storia: int = 60, ogni: int = 3,
           costi: float = 0.0015) -> RisultatoBacktest:
    """Esegue una strategia sui rendimenti mensili ``r``; ``costi`` per euro scambiato."""
    asset = list(r.columns)
    w = pd.Series(0.0, index=asset)
    investito = False
    rend, pesi, rot, cst = {}, {}, {}, {}
    for t in range(minimo_storia, len(r)):
        data = r.index[t]
        if not investito or (t - minimo_storia) % ogni == 0:
            storia = r.iloc[:t]
            nuovo = strategia(storia, w if investito else pd.Series(np.nan, index=asset))
            nuovo = nuovo.reindex(asset).fillna(0.0).clip(lower=0.0)
            nuovo /= nuovo.sum()
            scambi = (nuovo - w).abs().sum() if investito else 1.0  # il primo acquisto costa come un giro intero
            rot[data] = 0.5 * scambi if investito else 0.0
            cst[data] = costi * scambi
            pesi[data] = nuovo
            w, investito = nuovo, True
        else:
            cst[data] = 0.0
        lordo = float(w @ r.iloc[t])
        rend[data] = lordo - cst[data]
        w = w * (1 + r.iloc[t]) / (1 + lordo)
    return RisultatoBacktest(nome, pd.Series(rend, name=nome), pd.DataFrame(pesi).T, pd.Series(rot), pd.Series(cst))


# --------------------------------------------------------------------------- #
# Strategie
# --------------------------------------------------------------------------- #
def strategie_standard(pesi_mercato: pd.Series, vincoli: Vincoli, avversione: float = bl.AVVERSIONE_DEFAULT,
                       risk_free: float = 0.0, costi: float = 0.0015, rotazione_massima: float | None = 0.25,
                       alfa_cvar: float = 0.90, simulazioni: int = 100) -> dict[str, Strategia]:
    """Le strategie confrontate nel backtest, tutte con gli stessi vincoli del mandato."""
    mercato = pesi_mercato / pesi_mercato.sum()

    def s_mercato(storia, _):
        return mercato

    def s_uguali(storia, _):
        return pd.Series(1 / storia.shape[1], index=storia.columns)

    def s_media_varianza(storia, _):
        e = stima(storia)
        return Ottimizzatore(e.media, e.covarianza, vincoli).media_varianza(avversione).pesi

    def s_black_litterman(storia, attuali):
        e = stima(storia)
        ris = bl.modello(e.covarianza, mercato, bl.viste_momentum(storia), avversione, risk_free=risk_free)
        v = Vincoli(vincoli.minimo, vincoli.massimo, vincoli.gruppi)
        if attuali.notna().all():
            v.pesi_attuali, v.costi, v.rotazione_massima = attuali, costi, rotazione_massima
        try:
            return Ottimizzatore(ris.posteriore, ris.covarianza, v).media_varianza(avversione).pesi
        except RuntimeError:  # pesi derivati fuori mandato: serve più rotazione di quella concessa
            v.rotazione_massima = None
            return Ottimizzatore(ris.posteriore, ris.covarianza, v).media_varianza(avversione).pesi

    def s_hrp(storia, _):
        return hrp(stima(storia).covarianza).pesi

    def s_cvar(storia, _):
        return minimo_cvar(storia, alfa_cvar, vincoli).pesi

    def s_ricampionata(storia, _):
        e = stima(storia)
        return ricampiona(e.media, e.covarianza, len(storia), avversione, vincoli, simulazioni, seed=0).pesi

    return {
        "Mercato (riferimento)": s_mercato,
        "Pesi uguali": s_uguali,
        "Media-varianza storica": s_media_varianza,
        "Media-varianza ricampionata": s_ricampionata,
        "Black-Litterman": s_black_litterman,
        "HRP": s_hrp,
        "Minimo CVaR": s_cvar,
    }


def confronta(r: pd.DataFrame, strategie: dict[str, Strategia], minimo_storia: int = 60, ogni: int = 3,
              costi: float = 0.0015) -> dict[str, RisultatoBacktest]:
    return {nome: esegui(r, s, nome, minimo_storia, ogni, costi) for nome, s in strategie.items()}


def stabilita_pesi(ris: RisultatoBacktest) -> float:
    """Variazione media dei pesi tra un ribilanciamento e il successivo (più bassa = più stabile)."""
    p = ris.pesi
    return float(0.5 * p.diff().abs().sum(axis=1).iloc[1:].mean()) if len(p) > 1 else 0.0
