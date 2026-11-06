"""Analisi completa: dalle stime al portafoglio proposto, con il confronto tra i metodi e il backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest as bt
from . import black_litterman as bl
from . import metriche
from .cvar import RisultatoCVaR, cvar_empirico, frontiera_cvar, minimo_cvar
from .hrp import RisultatoHRP, hrp
from .ottimizzatore import Ottimizzatore, Portafoglio, Vincoli
from .ricampionamento import RisultatoRicampionamento, ricampiona
from .stime import Stima, stima

PROPOSTO = "Proposto (Black-Litterman)"


@dataclass
class Configurazione:
    """Tutto ciò che descrive il cliente e il mandato."""

    pesi_mercato: pd.Series
    pesi_attuali: pd.Series
    vincoli: Vincoli
    viste: list[bl.Vista] = field(default_factory=list)
    importo: float = 500_000.0
    risk_free: float = 0.025
    avversione: float = bl.AVVERSIONE_DEFAULT
    tau: float = bl.TAU_DEFAULT
    costi: float = 0.0015
    rotazione_massima: float | None = 0.30
    alfa: float = 0.95
    cliente: str = "Cliente"


@dataclass
class Analisi:
    config: Configurazione
    prezzi: pd.DataFrame
    rendimenti: pd.DataFrame
    stima: Stima
    bl: bl.RisultatoBL
    portafogli: dict[str, Portafoglio]
    hrp: RisultatoHRP
    cvar: RisultatoCVaR
    ricampionamento: RisultatoRicampionamento
    confronto: pd.DataFrame  # metriche ex ante ed ex post per portafoglio
    frontiere: dict[str, pd.DataFrame]
    frontiera_cvar: pd.DataFrame
    operazioni: pd.DataFrame
    backtest: dict[str, bt.RisultatoBacktest] = field(repr=False)
    metriche_backtest: pd.DataFrame = field(repr=False)

    @property
    def proposto(self) -> Portafoglio:
        return self.portafogli[PROPOSTO]


def _rendimenti_portafoglio(r: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Rendimenti mensili storici con i pesi ribilanciati ogni mese."""
    return r @ w.reindex(r.columns).fillna(0.0)


def tabella_confronto(portafogli: dict[str, Portafoglio], r: pd.DataFrame, bl_ris: bl.RisultatoBL,
                      cov: pd.DataFrame, risk_free: float, alfa: float) -> pd.DataFrame:
    """Le stesse misure per ogni portafoglio. I rendimenti attesi usano tutti le stime di Black-Litterman,
    così il confronto non premia chi ha ottimizzato proprio sulle medie storiche."""
    mu, s = bl_ris.posteriore, bl_ris.covarianza
    righe = {}
    for nome, p in portafogli.items():
        w = p.pesi.reindex(cov.index).fillna(0.0)
        atteso = float(mu @ w)
        vol = float(np.sqrt(w @ s @ w))
        storico = _rendimenti_portafoglio(r, w)
        righe[nome] = {
            "Rendimento atteso": atteso,
            "Volatilità attesa": vol,
            "Sharpe atteso": (atteso - risk_free) / vol,
            "Equivalente certo": atteso - 0.5 * bl_ris.avversione * vol**2,
            f"CVaR {alfa:.0%} mensile storico": cvar_empirico(storico.to_numpy(), alfa),
            "Massima perdita storica": metriche.drawdown(storico).min(),
            "Asset effettivi": metriche.numero_effettivo(w),
            "Rapporto di diversificazione": metriche.rapporto_diversificazione(w, cov),
            "Peso massimo": float(w.max()),
        }
    return pd.DataFrame(righe).T


def analizza(prezzi: pd.DataFrame, config: Configurazione, backtest: bool = True, simulazioni: int = 300,
             simulazioni_backtest: int = 100, seed: int = 0) -> Analisi:
    r = prezzi.pct_change().dropna()
    asset = list(r.columns)
    c = config
    e = stima(r)
    v = c.vincoli
    v_base = Vincoli(v.minimo, v.massimo, v.gruppi)

    ris_bl = bl.modello(e.covarianza, c.pesi_mercato, c.viste, c.avversione, c.tau, c.risk_free)
    v_prop = Vincoli(v.minimo, v.massimo, v.gruppi, c.pesi_attuali, c.costi, c.rotazione_massima)
    try:
        proposto = Ottimizzatore(ris_bl.posteriore, ris_bl.covarianza, v_prop).media_varianza(c.avversione, PROPOSTO)
    except RuntimeError:
        v_prop.rotazione_massima = None  # il portafoglio attuale è troppo lontano dal mandato
        proposto = Ottimizzatore(ris_bl.posteriore, ris_bl.covarianza, v_prop).media_varianza(c.avversione, PROPOSTO)

    storico = Ottimizzatore(e.media, e.covarianza, v_base).media_varianza(c.avversione, "Media-varianza storica")
    ric = ricampiona(e.media, e.covarianza, len(r), c.avversione, v_base, simulazioni, seed)
    ris_hrp = hrp(e.covarianza)
    ris_cvar = minimo_cvar(r, c.alfa, v_base)

    def da_pesi(nome: str, w: pd.Series) -> Portafoglio:
        w = w.reindex(asset).fillna(0.0)
        w0 = c.pesi_attuali.reindex(asset).fillna(0.0)
        rot = 0.5 * float((w - w0).abs().sum())
        return Portafoglio(nome, w.rename(nome), float(ris_bl.posteriore @ w),
                           float(np.sqrt(w @ ris_bl.covarianza @ w)), rot, 2 * rot * c.costi)

    portafogli = {
        "Attuale": da_pesi("Attuale", c.pesi_attuali),
        "Mercato": da_pesi("Mercato", c.pesi_mercato / c.pesi_mercato.sum()),
        "Media-varianza storica": da_pesi("Media-varianza storica", storico.pesi),
        "Media-varianza ricampionata": da_pesi("Media-varianza ricampionata", ric.pesi),
        "HRP": da_pesi("HRP", ris_hrp.pesi),
        "Minimo CVaR": da_pesi("Minimo CVaR", ris_cvar.pesi),
        PROPOSTO: da_pesi(PROPOSTO, proposto.pesi),
    }
    confronto = tabella_confronto(portafogli, r, ris_bl, e.covarianza, c.risk_free, c.alfa)

    frontiere = {
        "Stime storiche": Ottimizzatore(e.media, e.covarianza, v_base).frontiera(25),
        "Black-Litterman": Ottimizzatore(ris_bl.posteriore, ris_bl.covarianza, v_base).frontiera(25),
    }
    fr_cvar = frontiera_cvar(r, c.alfa, v_base, 15)

    w0 = c.pesi_attuali.reindex(asset).fillna(0.0)
    w1 = portafogli[PROPOSTO].pesi
    operazioni = pd.DataFrame({
        "Peso attuale": w0, "Peso proposto": w1, "Variazione": w1 - w0,
        "Controvalore attuale": w0 * c.importo, "Controvalore proposto": w1 * c.importo,
        "Operazione": (w1 - w0) * c.importo,
    })
    operazioni.loc[operazioni["Operazione"].abs() < 1, "Operazione"] = 0.0

    risultati_bt: dict[str, bt.RisultatoBacktest] = {}
    metr = pd.DataFrame()
    if backtest and len(r) >= 84:
        strategie = bt.strategie_standard(c.pesi_mercato, v_base, c.avversione, c.risk_free, c.costi,
                                          rotazione_massima=0.25, simulazioni=simulazioni_backtest)
        risultati_bt = bt.confronta(r, strategie, minimo_storia=60, ogni=3, costi=c.costi)
        metr = pd.DataFrame({
            n: {**metriche.tabella(x.rendimenti, c.risk_free, 0.95), "Rotazione annua": x.rotazione_annua,
                "Costi annui": float(x.costi.sum() / (len(x.rendimenti) / 12)),
                "Instabilità dei pesi": bt.stabilita_pesi(x)}
            for n, x in risultati_bt.items()
        }).T
    return Analisi(c, prezzi, r, e, ris_bl, portafogli, ris_hrp, ris_cvar, ric, confronto, frontiere, fr_cvar,
                   operazioni, risultati_bt, metr)
