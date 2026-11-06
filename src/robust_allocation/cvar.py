"""Portafoglio a minimo CVaR con la programmazione lineare di Rockafellar e Uryasev (2000).

Il CVaR (Expected Shortfall) è la perdita media nei casi peggiori oltre il VaR. Su un insieme di scenari
r₁ … r_T la sua minimizzazione diventa un problema lineare:

    min  z + 1/((1 − α) T) · Σ uₜ
    con  uₜ ≥ −rₜ'w − z,  uₜ ≥ 0,  Σ w = 1,  vincoli sui pesi

dove all'ottimo z è il VaR. A differenza della varianza, il CVaR guarda solo alle perdite e coglie le code
pesanti e le correlazioni che salgono nelle crisi. Risolto con HiGHS (``scipy.optimize.linprog``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .ottimizzatore import Vincoli


@dataclass
class RisultatoCVaR:
    pesi: pd.Series
    cvar: float  # perdita media nella coda, per periodo, positiva se perdita
    var: float
    rendimento: float  # medio per periodo sugli scenari


def cvar_empirico(rendimenti: np.ndarray, alfa: float = 0.95) -> float:
    """CVaR di una serie di rendimenti: media delle perdite nella coda (1 − α), con la frazione di bordo."""
    perdite = np.sort(-np.asarray(rendimenti, dtype=float))[::-1]
    t = len(perdite)
    k = (1 - alfa) * t
    interi = int(np.floor(k))
    somma = perdite[:interi].sum() + (k - interi) * (perdite[interi] if interi < t else 0.0)
    return float(somma / k)


def minimo_cvar(scenari: pd.DataFrame, alfa: float = 0.95, vincoli: Vincoli | None = None,
                rendimento_minimo: float | None = None) -> RisultatoCVaR:
    """Pesi che minimizzano il CVaR sugli scenari; ``rendimento_minimo`` è per periodo, come gli scenari."""
    vincoli = vincoli or Vincoli()
    asset = list(scenari.columns)
    r = scenari.to_numpy(dtype=float)
    t, n = r.shape
    # Variabili: w (n), z (1), u (t)
    c = np.r_[np.zeros(n), 1.0, np.full(t, 1 / ((1 - alfa) * t))]
    a_ub = [np.hstack([-r, -np.ones((t, 1)), -np.eye(t)])]
    b_ub = [np.zeros(t)]
    for nome, (membri, gmin, gmax) in vincoli.gruppi.items():
        m = np.array([a in membri for a in asset], dtype=float)
        if not m.any():
            raise KeyError(f"Il gruppo {nome} non contiene asset dell'universo")
        riga = np.r_[m, 0.0, np.zeros(t)]
        a_ub += [riga[None, :], -riga[None, :]]
        b_ub += [np.array([gmax]), np.array([-gmin])]
    if rendimento_minimo is not None:
        a_ub.append(np.r_[-r.mean(axis=0), 0.0, np.zeros(t)][None, :])
        b_ub.append(np.array([-rendimento_minimo]))
    a_eq = np.r_[np.ones(n), 0.0, np.zeros(t)][None, :]
    limiti = vincoli.limiti(asset) + [(None, None)] + [(0, None)] * t
    ris = linprog(c, A_ub=np.vstack(a_ub), b_ub=np.concatenate(b_ub), A_eq=a_eq, b_eq=[1.0], bounds=limiti,
                  method="highs")
    if not ris.success:
        raise RuntimeError(f"Minimo CVaR non riuscito: {ris.message}")
    w = np.where(np.abs(ris.x[:n]) < 1e-9, 0.0, ris.x[:n])
    w /= w.sum()
    return RisultatoCVaR(pd.Series(w, index=asset, name="Minimo CVaR"), float(ris.fun), float(ris.x[n]),
                         float(r.mean(axis=0) @ w))


def frontiera_cvar(scenari: pd.DataFrame, alfa: float = 0.95, vincoli: Vincoli | None = None,
                   punti: int = 20) -> pd.DataFrame:
    """Frontiera rendimento-CVaR: CVaR minimo per una griglia di rendimenti medi raggiungibili."""
    base = minimo_cvar(scenari, alfa, vincoli)
    medie = scenari.mean()
    vincoli = vincoli or Vincoli()
    limiti = vincoli.limiti(list(scenari.columns))
    # Rendimento massimo raggiungibile rispettando i vincoli: un problema lineare.
    n = len(medie)
    a_ub, b_ub = [], []
    for membri, gmin, gmax in vincoli.gruppi.values():
        m = np.array([a in membri for a in scenari.columns], dtype=float)
        a_ub += [m, -m]
        b_ub += [gmax, -gmin]
    mx = linprog(-medie.to_numpy(), A_ub=np.array(a_ub) if a_ub else None, b_ub=b_ub or None,
                 A_eq=np.ones((1, n)), b_eq=[1.0], bounds=limiti, method="highs")
    alto = -mx.fun if mx.success else medie.max()
    righe = []
    for target in np.linspace(base.rendimento, alto * 0.999, punti):
        try:
            p = minimo_cvar(scenari, alfa, vincoli, rendimento_minimo=target)
        except RuntimeError:
            continue
        righe.append({"Rendimento": p.rendimento, "CVaR": p.cvar, **p.pesi.to_dict()})
    return pd.DataFrame(righe)
