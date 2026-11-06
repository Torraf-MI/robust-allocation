"""Ottimizzazione media-varianza con i vincoli di un mandato reale: pesi, classi di attività, rotazione e costi.

Il problema è risolto con SLSQP (``scipy.optimize``) e gradienti analitici. Quando si parte da un portafoglio
esistente, le operazioni sono variabili esplicite (acquisti ``b`` e vendite ``s``, entrambe non negative): così il
costo proporzionale ``c · Σ(b + s)`` e il limite di rotazione restano lineari e il problema resta convesso.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Vincoli:
    """Vincoli del mandato.

    - ``minimo`` / ``massimo``: peso per asset, un numero per tutti o un dizionario per asset;
    - ``gruppi``: ``{"Azionario": (["Azionario USA", ...], 0.30, 0.70)}``;
    - ``pesi_attuali``: portafoglio di partenza (serve per costi e rotazione);
    - ``costi``: costo proporzionale di ogni euro scambiato (0,0015 = 15 punti base);
    - ``rotazione_massima``: metà della somma delle variazioni dei pesi (0,20 = si sposta al massimo il 20%).
    """

    minimo: float | dict[str, float] = 0.0
    massimo: float | dict[str, float] = 1.0
    gruppi: dict[str, tuple[list[str], float, float]] = field(default_factory=dict)
    pesi_attuali: pd.Series | None = None
    costi: float = 0.0
    rotazione_massima: float | None = None

    def limiti(self, asset: list[str]) -> list[tuple[float, float]]:
        def valore(v, a, default):
            return float(v.get(a, default)) if isinstance(v, dict) else float(v)
        return [(valore(self.minimo, a, 0.0), valore(self.massimo, a, 1.0)) for a in asset]


@dataclass
class Portafoglio:
    nome: str
    pesi: pd.Series
    rendimento: float  # atteso, annuo
    volatilita: float
    rotazione: float = 0.0
    costi: float = 0.0  # costo stimato del passaggio dai pesi attuali, in % del patrimonio

    def sharpe(self, risk_free: float = 0.0) -> float:
        return (self.rendimento - risk_free) / self.volatilita if self.volatilita > 0 else np.nan


class Ottimizzatore:
    def __init__(self, mu: pd.Series, cov: pd.DataFrame, vincoli: Vincoli | None = None):
        self.asset = list(cov.index)
        self.mu = mu.reindex(self.asset).to_numpy(dtype=float)
        self.cov = cov.loc[self.asset, self.asset].to_numpy(dtype=float)
        self.vincoli = vincoli or Vincoli()
        self.n = len(self.asset)
        w0 = self.vincoli.pesi_attuali
        self.w0 = None if w0 is None else w0.reindex(self.asset).fillna(0.0).to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    def _impostazione(self):
        """Variabili, limiti e vincoli in forma SLSQP. Con un portafoglio di partenza: x = [w, b, s]."""
        n = self.n
        con_operazioni = self.w0 is not None
        limiti = self.vincoli.limiti(self.asset)
        if con_operazioni:
            limiti = limiti + [(0.0, 1.0)] * (2 * n)
        dim = 3 * n if con_operazioni else n

        def w_di(x):
            return x[:n]

        cons = [{"type": "eq", "fun": lambda x: w_di(x).sum() - 1.0,
                 "jac": lambda x: np.r_[np.ones(n), np.zeros(dim - n)]}]
        for nome, (membri, gmin, gmax) in self.vincoli.gruppi.items():
            maschera = np.array([a in membri for a in self.asset], dtype=float)
            if not maschera.any():
                raise KeyError(f"Il gruppo {nome} non contiene asset dell'universo")
            j = np.r_[maschera, np.zeros(dim - n)]
            cons.append({"type": "ineq", "fun": lambda x, m=maschera, lo=gmin: m @ x[:n] - lo,
                         "jac": lambda x, j=j: j})
            cons.append({"type": "ineq", "fun": lambda x, m=maschera, hi=gmax: hi - m @ x[:n],
                         "jac": lambda x, j=j: -j})
        if con_operazioni:
            w0 = self.w0
            jac_bilancio = np.hstack([np.eye(n), -np.eye(n), np.eye(n)])
            cons.append({"type": "eq", "fun": lambda x: x[:n] - w0 - x[n:2 * n] + x[2 * n:],
                         "jac": lambda x: jac_bilancio})
            if self.vincoli.rotazione_massima is not None:
                jr = np.r_[np.zeros(n), -0.5 * np.ones(2 * n)]
                cons.append({"type": "ineq", "fun": lambda x: self.vincoli.rotazione_massima - 0.5 * x[n:].sum(),
                             "jac": lambda x: jr})
        return dim, limiti, cons, con_operazioni

    def _punto_iniziale(self, dim: int, limiti, con_operazioni: bool) -> np.ndarray:
        n = self.n
        lo = np.array([b[0] for b in limiti[:n]])
        hi = np.array([b[1] for b in limiti[:n]])
        if con_operazioni:
            w = np.clip(self.w0, lo, hi)
        else:
            w = np.clip(np.full(n, 1 / n), lo, hi)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
        if not con_operazioni:
            return w
        d = w - self.w0
        return np.r_[w, np.maximum(d, 0), np.maximum(-d, 0)]

    def _risolvi(self, obiettivo, gradiente, nome: str) -> Portafoglio:
        dim, limiti, cons, con_operazioni = self._impostazione()
        x0 = self._punto_iniziale(dim, limiti, con_operazioni)
        ris = minimize(obiettivo, x0, jac=gradiente, method="SLSQP", bounds=limiti, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-12})
        if not ris.success:
            raise RuntimeError(f"Ottimizzazione non riuscita ({nome}): {ris.message}")
        return self._portafoglio(ris.x[:self.n], nome)

    def _portafoglio(self, w: np.ndarray, nome: str) -> Portafoglio:
        w = np.where(np.abs(w) < 1e-8, 0.0, w)
        w = w / w.sum()
        rot = 0.5 * np.abs(w - self.w0).sum() if self.w0 is not None else 0.0
        return Portafoglio(nome, pd.Series(w, index=self.asset, name=nome), float(self.mu @ w),
                           float(np.sqrt(w @ self.cov @ w)), rot, 2 * rot * self.vincoli.costi)

    def _penale_costi(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.w0 is None or self.vincoli.costi == 0:
            return 0.0, np.zeros_like(x)
        n = self.n
        c = self.vincoli.costi
        g = np.r_[np.zeros(n), c * np.ones(2 * n)]
        return c * x[n:].sum(), g

    # ------------------------------------------------------------------ #
    def media_varianza(self, avversione: float, nome: str = "Media-varianza") -> Portafoglio:
        """Massimizza μ'w − δ/2 · w'Σw − costi."""
        n = self.n

        def f(x):
            w = x[:n]
            pen, _ = self._penale_costi(x)
            return -(self.mu @ w - 0.5 * avversione * w @ self.cov @ w) + pen

        def g(x):
            w = x[:n]
            _, gp = self._penale_costi(x)
            gw = -(self.mu - avversione * self.cov @ w)
            return np.r_[gw, np.zeros(len(x) - n)] + gp

        return self._risolvi(f, g, nome)

    def minima_varianza(self, rendimento_minimo: float | None = None, nome: str = "Minima varianza") -> Portafoglio:
        """Minimizza la varianza (più i costi), eventualmente con un rendimento atteso minimo."""
        n = self.n
        extra = []
        if rendimento_minimo is not None:
            extra.append(rendimento_minimo)

        def f(x):
            w = x[:n]
            pen, _ = self._penale_costi(x)
            return 0.5 * w @ self.cov @ w + pen

        def g(x):
            _, gp = self._penale_costi(x)
            return np.r_[self.cov @ x[:n], np.zeros(len(x) - n)] + gp

        if not extra:
            return self._risolvi(f, g, nome)
        dim, limiti, cons, con_operazioni = self._impostazione()
        j = np.r_[self.mu, np.zeros(dim - n)]
        cons.append({"type": "ineq", "fun": lambda x: self.mu @ x[:n] - rendimento_minimo, "jac": lambda x: j})
        x0 = self._punto_iniziale(dim, limiti, con_operazioni)
        ris = minimize(f, x0, jac=g, method="SLSQP", bounds=limiti, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-12})
        if not ris.success:
            raise RuntimeError(f"Ottimizzazione non riuscita ({nome}): {ris.message}")
        return self._portafoglio(ris.x[:n], nome)

    def frontiera(self, punti: int = 30) -> pd.DataFrame:
        """Frontiera efficiente vincolata: volatilità minima per una griglia di rendimenti raggiungibili."""
        base = Ottimizzatore(pd.Series(self.mu, index=self.asset), pd.DataFrame(self.cov, self.asset, self.asset),
                             Vincoli(self.vincoli.minimo, self.vincoli.massimo, self.vincoli.gruppi))
        w_min = base.minima_varianza()
        w_max = base.media_varianza(avversione=1e-4)
        righe = []
        for target in np.linspace(w_min.rendimento, w_max.rendimento, punti):
            try:
                p = base.minima_varianza(rendimento_minimo=target)
            except RuntimeError:
                continue
            righe.append({"Rendimento": p.rendimento, "Volatilità": p.volatilita, **p.pesi.to_dict()})
        return pd.DataFrame(righe).drop_duplicates(subset=["Rendimento"]).reset_index(drop=True)


def soluzione_analitica(mu: np.ndarray, cov: np.ndarray, avversione: float) -> np.ndarray:
    """Pesi media-varianza con il solo vincolo di bilancio (somma 1), in forma chiusa. Usata nei test."""
    inv = np.linalg.inv(cov)
    uno = np.ones(len(mu))
    gamma = (uno @ inv @ mu - avversione) / (uno @ inv @ uno)
    return inv @ (mu - gamma * uno) / avversione
