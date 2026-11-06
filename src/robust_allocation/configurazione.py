"""Configurazione del cliente: esempio dimostrativo e lettura da file JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import dati
from .analisi import Configurazione
from .black_litterman import Vista
from .ottimizzatore import Vincoli

PESI_ATTUALI_ESEMPIO = pd.Series({
    "Governativi EUR": 0.35, "Corporate EUR": 0.15, "High Yield": 0.05,
    "Azionario Europa": 0.25, "Azionario USA": 0.15, "Oro": 0.05,
})

VISTE_ESEMPIO = [
    Vista("relativa", ["Azionario Europa"], 0.015, 0.40, ["Azionario USA"],
          "L'azionario europeo farà meglio di quello USA di 1,5% all'anno (valutazioni più basse)"),
    Vista("assoluta", ["Governativi EUR"], 0.032, 0.60, descrizione=(
          "I governativi EUR renderanno il 3,2% all'anno (rendimento a scadenza attuale)")),
    Vista("relativa", ["Azionario Emergenti"], 0.020, 0.30, ["Azionario Giappone"],
          "Gli emergenti faranno meglio del Giappone di 2% all'anno"),
]


def vincoli_esempio(asset: list[str]) -> Vincoli:
    g = dati.gruppi_per(asset)
    gruppi = {}
    for nome, lo, hi in (("Azionario", 0.30, 0.60), ("Obbligazionario", 0.30, 0.60), ("Alternativi", 0.0, 0.15)):
        if nome in g:
            gruppi[nome] = (g[nome], lo, hi)
    return Vincoli(minimo=0.0, massimo=0.30, gruppi=gruppi)


def esempio(cliente: str = "Cliente", importo: float = 500_000.0) -> Configurazione:
    return Configurazione(
        pesi_mercato=dati.PESI_MERCATO.copy(),
        pesi_attuali=PESI_ATTUALI_ESEMPIO.reindex(dati.ASSET).fillna(0.0),
        vincoli=vincoli_esempio(dati.ASSET),
        viste=list(VISTE_ESEMPIO),
        importo=importo,
        cliente=cliente,
    )


def carica(percorso: str | Path) -> tuple[pd.DataFrame, Configurazione]:
    """Legge un file JSON con prezzi (percorso CSV relativo al file), mandato, view e portafoglio attuale."""
    percorso = Path(percorso)
    d = json.loads(percorso.read_text(encoding="utf-8"))
    if "prezzi" in d:
        prezzi = dati.carica_csv(percorso.parent / d["prezzi"])
    else:
        prezzi = dati.prezzi_dimostrativi()
    asset = list(prezzi.columns)
    v = d.get("vincoli", {})
    gruppi = {n: (g["asset"], float(g.get("min", 0.0)), float(g.get("max", 1.0)))
              for n, g in v.get("gruppi", {}).items()}
    mercato = pd.Series(d.get("pesi_mercato", {})).reindex(asset).fillna(0.0)
    if mercato.sum() == 0:
        mercato = pd.Series(1 / len(asset), index=asset)
    attuali = pd.Series(d["pesi_attuali"]).reindex(asset).fillna(0.0)
    cfg = Configurazione(
        pesi_mercato=mercato / mercato.sum(),
        pesi_attuali=attuali / attuali.sum(),
        vincoli=Vincoli(v.get("minimo", 0.0), v.get("massimo", 1.0), gruppi),
        viste=[Vista.da_dict(x) for x in d.get("viste", [])],
        importo=float(d.get("importo", 500_000)),
        risk_free=float(d.get("risk_free", 0.025)),
        avversione=float(d.get("avversione", 2.5)),
        tau=float(d.get("tau", 0.05)),
        costi=float(d.get("costi", 0.0015)),
        rotazione_massima=d.get("rotazione_massima", 0.30),
        alfa=float(d.get("alfa", 0.95)),
        cliente=d.get("cliente", "Cliente"),
    )
    return prezzi, cfg
