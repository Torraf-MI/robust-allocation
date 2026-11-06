"""Dati: prezzi da CSV, rendimenti mensili, universo dimostrativo con regimi di crisi, pesi di mercato.

I dati dimostrativi sono **simulati** con quattro regimi (calma, crisi, crollo, shock inflazionistico) collocati
nelle date delle crisi reali, così il backtest attraversa episodi simili al 2008, al 2011, al 2020 e al 2022.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASSET = [
    "Azionario USA",
    "Azionario Europa",
    "Azionario Giappone",
    "Azionario Emergenti",
    "Governativi EUR",
    "Corporate EUR",
    "High Yield",
    "Oro",
    "Immobiliare quotato",
]

CLASSI = {
    "Azionario": ["Azionario USA", "Azionario Europa", "Azionario Giappone", "Azionario Emergenti"],
    "Obbligazionario": ["Governativi EUR", "Corporate EUR", "High Yield"],
    "Alternativi": ["Oro", "Immobiliare quotato"],
}

# Pesi di un portafoglio di mercato globale multi-asset (ordine di grandezza delle capitalizzazioni):
# sono il punto di partenza "neutrale" del modello di Black-Litterman.
PESI_MERCATO = pd.Series(
    [0.30, 0.09, 0.04, 0.06, 0.22, 0.12, 0.05, 0.07, 0.05], index=ASSET, name="Mercato"
)

# Esposizioni ai fattori [azioni, tassi, credito, oro] usate per costruire le correlazioni.
_ESPOSIZIONI = np.array([
    [1.00, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.10, 0.00],
    [0.80, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.20, 0.00],
    [-0.10, 1.00, 0.00, 0.00],
    [0.10, 0.80, 0.50, 0.00],
    [0.45, 0.20, 0.80, 0.00],
    [0.05, 0.25, 0.00, 1.00],
    [0.80, 0.30, 0.20, 0.00],
])

# Rendimenti e volatilità annui per regime.
_REGIMI = {
    "calma": {
        "mu": [0.125, 0.100, 0.080, 0.115, 0.030, 0.040, 0.070, 0.075, 0.100],
        "vol": [0.140, 0.150, 0.160, 0.190, 0.045, 0.040, 0.070, 0.150, 0.160],
        "idio": 0.45,
        "segno_gov": 1.0,
    },
    "crisi": {
        "mu": [-0.300, -0.340, -0.260, -0.380, 0.080, -0.040, -0.220, 0.120, -0.360],
        "vol": [0.270, 0.300, 0.260, 0.330, 0.070, 0.080, 0.180, 0.220, 0.320],
        "idio": 0.15,
        "segno_gov": 3.0,
    },
    "crollo": {
        "mu": [-0.900, -1.000, -0.750, -1.100, 0.150, -0.200, -0.650, 0.050, -1.100],
        "vol": [0.300, 0.330, 0.280, 0.360, 0.080, 0.100, 0.220, 0.250, 0.380],
        "idio": 0.10,
        "segno_gov": 3.0,
    },
    "inflazione": {
        "mu": [-0.180, -0.150, -0.080, -0.200, -0.170, -0.150, -0.130, 0.000, -0.250],
        "vol": [0.220, 0.230, 0.180, 0.220, 0.090, 0.080, 0.120, 0.160, 0.250],
        "idio": 0.25,
        "segno_gov": -3.0,
    },
}

CRISI = [
    ("2002-04", "2002-09", "crisi", "Bolla tecnologica"),
    ("2007-11", "2008-08", "crisi", "Crisi dei mutui subprime"),
    ("2008-09", "2009-02", "crollo", "Fallimento di Lehman Brothers"),
    ("2011-07", "2011-11", "crisi", "Crisi del debito sovrano"),
    ("2020-02", "2020-03", "crollo", "Pandemia"),
    ("2022-01", "2022-10", "inflazione", "Shock inflazionistico e rialzo dei tassi"),
]


def _correlazione(regime: dict) -> np.ndarray:
    b = _ESPOSIZIONI.copy()
    # In crisi le obbligazioni governative diventano un rifugio (correlazione negativa con le azioni);
    # nello shock inflazionistico azioni e obbligazioni scendono insieme.
    b[4, 0] *= regime["segno_gov"]
    comune = b @ b.T
    cov = comune + np.diag(np.diag(comune) * regime["idio"] + 0.05)
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def regimi_dimostrativi(date: pd.DatetimeIndex) -> pd.Series:
    """Regime di ogni mese: calma, salvo i periodi in ``CRISI``."""
    reg = pd.Series("calma", index=date)
    for inizio, fine, tipo, _ in CRISI:
        reg[(date >= pd.Timestamp(inizio)) & (date <= pd.Timestamp(fine) + pd.offsets.MonthEnd(0))] = tipo
    return reg


def prezzi_dimostrativi(inizio: str = "2001-12-31", fine: str = "2024-12-31", seed: int = 399,
                        gradi_liberta: float = 6.0) -> pd.DataFrame:
    """Prezzi mensili simulati (base 100) con innovazioni a code pesanti (t di Student)."""
    date = pd.date_range(inizio, fine, freq="ME")
    regimi = regimi_dimostrativi(date[1:])
    rng = np.random.default_rng(seed)
    n = len(ASSET)
    fattori = {k: np.linalg.cholesky(_correlazione(v)) for k, v in _REGIMI.items()}
    righe = []
    for reg in regimi:
        p = _REGIMI[reg]
        z = rng.standard_normal(n)
        # Una t multivariata: stesso fattore di scala per tutti gli asset, così le code arrivano insieme.
        scala = np.sqrt((gradi_liberta - 2) / rng.chisquare(gradi_liberta))
        e = fattori[reg] @ z * scala
        vol = np.asarray(p["vol"]) / np.sqrt(12)
        mu = np.asarray(p["mu"]) / 12 - 0.5 * vol**2
        righe.append(np.expm1(mu + vol * e))
    r = pd.DataFrame(righe, index=date[1:], columns=ASSET)
    prezzi = 100 * (1 + r).cumprod()
    prezzi.loc[date[0]] = 100.0
    prezzi = prezzi.sort_index().round(4)
    prezzi.index.name = "Data"
    return prezzi


def carica_csv(percorso: str | Path) -> pd.DataFrame:
    """Prezzi da CSV: una colonna di date e una colonna per strumento. Converte in fine mese se giornalieri."""
    df = pd.read_csv(percorso, index_col=0, parse_dates=True).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    passo = df.index.to_series().diff().median()
    if passo is not pd.NaT and passo < pd.Timedelta(days=20):
        df = df.resample("ME").last()
    return df.dropna()


def rendimenti(prezzi: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti semplici per periodo."""
    return prezzi.pct_change().dropna(how="all").dropna()


def gruppi_per(asset: list[str]) -> dict[str, list[str]]:
    """Classi di attività ristrette agli asset presenti."""
    return {k: [a for a in v if a in asset] for k, v in CLASSI.items() if any(a in asset for a in v)}
