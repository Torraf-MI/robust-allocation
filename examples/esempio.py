"""Uso come libreria: view del consulente, Black-Litterman con vincoli e costi, HRP e minimo CVaR."""

import pandas as pd

from robust_allocation import Ottimizzatore, Vincoli, Vista, hrp, minimo_cvar, modello, stima
from robust_allocation.dati import PESI_MERCATO, carica_csv, rendimenti

r = rendimenti(carica_csv("data/prezzi_esempio.csv"))
e = stima(r)  # media storica e covarianza con shrinkage di Ledoit-Wolf, annualizzate
print(f"Intensità dello shrinkage: {e.intensita:.0%}")

viste = [
    Vista("relativa", ["Azionario Europa"], 0.015, fiducia=0.4, corti=["Azionario USA"]),
    Vista("assoluta", ["Governativi EUR"], 0.032, fiducia=0.6),
]
bl = modello(e.covarianza, PESI_MERCATO, viste, avversione=2.5, risk_free=0.025)
print(bl.tabella_viste().round(4).to_string(index=False))

attuale = pd.Series({"Governativi EUR": 0.35, "Corporate EUR": 0.15, "High Yield": 0.05,
                     "Azionario Europa": 0.25, "Azionario USA": 0.15, "Oro": 0.05}).reindex(r.columns).fillna(0)
azionario = ["Azionario USA", "Azionario Europa", "Azionario Giappone", "Azionario Emergenti"]
vincoli = Vincoli(massimo=0.30, gruppi={"Azionario": (azionario, 0.30, 0.60)},
                  pesi_attuali=attuale, costi=0.0015, rotazione_massima=0.30)
proposto = Ottimizzatore(bl.posteriore, bl.covarianza, vincoli).media_varianza(avversione=2.5)

confronto = pd.DataFrame({
    "Attuale": attuale,
    "Black-Litterman": proposto.pesi,
    "HRP": hrp(e.covarianza).pesi,
    "Minimo CVaR 95%": minimo_cvar(r, 0.95, Vincoli(massimo=0.30)).pesi,
})
print(confronto.round(3))
print(f"Rotazione {proposto.rotazione:.1%}, costi {proposto.costi:.3%} del patrimonio")
