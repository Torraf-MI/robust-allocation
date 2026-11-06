"""Grafici interattivi (plotly) e report HTML per il cliente."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

from . import metriche
from .analisi import PROPOSTO, Analisi

BLU, ARANCIO, ACQUA, VIOLA = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
BLU_SCURO, ROSSO, GIALLO, GRIGIO = "#1c5cab", "#e34948", "#e8a317", "#8c8b86"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
COLORI_ASSET = ["#1c5cab", "#2a78d6", "#7fb0ea", "#8a5cd1", "#1baf7a", "#78cfae", "#e8a317", "#c9a227", "#eb6834"]
COLORI_STRATEGIE = {
    "Mercato (riferimento)": GRIGIO, "Mercato": GRIGIO, "Attuale": "#b5b3ad", "Pesi uguali": "#c9a227",
    "Media-varianza storica": ROSSO, "Media-varianza ricampionata": ARANCIO, "Black-Litterman": BLU_SCURO,
    PROPOSTO: BLU_SCURO, "HRP": ACQUA, "Minimo CVaR": VIOLA,
}
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def _layout(fig: go.Figure, titolo: str, altezza: int = 430) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza,
        template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO},
        legend={"orientation": "h", "y": -0.2, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


def _etichette(fig: go.Figure, margine: int = 120) -> go.Figure:
    """Etichette dell'asse x inclinate, con spazio sufficiente sotto il grafico."""
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(margin={"b": margine}, legend={"y": -0.32})
    return fig


def _colore_asset(a: Analisi) -> dict[str, str]:
    return {x: COLORI_ASSET[i % len(COLORI_ASSET)] for i, x in enumerate(a.rendimenti.columns)}


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def grafico_rendimenti(a: Analisi) -> go.Figure:
    asset = list(a.rendimenti.columns)
    serie = [("Media storica", a.stima.media, "#b5b3ad"), ("Equilibrio di mercato", a.bl.equilibrio, ACQUA),
             ("Black-Litterman (con le view)", a.bl.posteriore, BLU_SCURO)]
    fig = go.Figure([go.Bar(name=n, x=asset, y=s.reindex(asset), marker_color=c,
                            hovertemplate="%{x}: %{y:.2%}<extra>" + n + "</extra>") for n, s, c in serie])
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title="Rendimento atteso annuo")
    return _etichette(_layout(fig, "Rendimenti attesi: storia, equilibrio e view del consulente", 480))


def grafico_allocazioni(a: Analisi) -> go.Figure:
    nomi = list(a.portafogli)
    col = _colore_asset(a)
    fig = go.Figure([go.Bar(name=x, x=nomi, y=[a.portafogli[n].pesi[x] for n in nomi], marker_color=col[x],
                            hovertemplate="%{x}<br>" + x + ": %{y:.1%}<extra></extra>")
                     for x in a.rendimenti.columns])
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(tickangle=0, ticktext=[n.replace(" ", "<br>", 1) for n in nomi], tickvals=nomi)
    return _layout(fig, "Allocazioni a confronto", 500)


def grafico_contributi(a: Analisi) -> go.Figure:
    nomi = list(a.portafogli)
    col = _colore_asset(a)
    contr = {n: metriche.contributi_rischio(a.portafogli[n].pesi, a.stima.covarianza) for n in nomi}
    fig = go.Figure([go.Bar(name=x, x=nomi, y=[contr[n][x] for n in nomi], marker_color=col[x],
                            hovertemplate="%{x}<br>" + x + ": %{y:.1%} del rischio<extra></extra>")
                     for x in a.rendimenti.columns])
    fig.update_layout(barmode="relative")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=0, ticktext=[n.replace(" ", "<br>", 1) for n in nomi], tickvals=nomi)
    return _layout(fig, "Da dove viene il rischio: quota della varianza per asset", 480)


def grafico_frontiere(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for nome, fr, colore in (("Frontiera con le stime storiche", a.frontiere["Stime storiche"], ROSSO),
                             ("Frontiera Black-Litterman", a.frontiere["Black-Litterman"], BLU_SCURO)):
        fig.add_trace(go.Scatter(x=fr["Volatilità"], y=fr["Rendimento"], mode="lines", name=nome,
                                 line={"color": colore, "width": 2.5},
                                 hovertemplate="Volatilità %{x:.1%}<br>Rendimento %{y:.2%}<extra></extra>"))
    for nome, p in a.portafogli.items():
        fig.add_trace(go.Scatter(x=[p.volatilita], y=[p.rendimento], mode="markers", name=nome,
                                 marker={"size": 13 if nome == PROPOSTO else 10,
                                         "color": COLORI_STRATEGIE.get(nome, GRIGIO),
                                         "symbol": "star" if nome == PROPOSTO else "circle",
                                         "line": {"color": "white", "width": 1}},
                                 hovertemplate=nome + "<br>Volatilità %{x:.1%}<br>Rendimento %{y:.2%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="Volatilità annua")
    fig.update_yaxes(tickformat=".1%", title="Rendimento atteso annuo")
    return _layout(fig, "Frontiere efficienti (i portafogli sono misurati con le stime Black-Litterman)", 500)


def grafico_instabilita(a: Analisi) -> go.Figure:
    sim = a.ricampionamento.simulazioni
    storico = a.portafogli["Media-varianza storica"].pesi
    fig = go.Figure()
    for x in sim.columns:
        fig.add_trace(go.Box(y=sim[x], name=x, marker_color=ARANCIO, boxpoints=False, showlegend=False,
                             hovertemplate=x + ": %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(sim.columns), y=storico.reindex(sim.columns), mode="markers",
                             name="Ottimo con le stime storiche", marker={"color": ROSSO, "size": 10, "symbol": "x"}))
    fig.add_trace(go.Scatter(x=list(sim.columns), y=a.ricampionamento.pesi.reindex(sim.columns), mode="markers",
                             name="Media delle simulazioni (ricampionato)",
                             marker={"color": BLU_SCURO, "size": 10, "symbol": "diamond"}))
    fig.update_yaxes(tickformat=".0%", title="Peso ottimo")
    fig.update_xaxes(tickangle=-35)
    fig = _layout(fig, f"Quanto è fragile la media-varianza: pesi ottimi in {len(sim)} storie simulate", 520)
    fig.update_layout(margin={"b": 130}, legend={"y": -0.38})
    return fig


def grafico_dendrogramma(a: Analisi) -> go.Figure:
    h = a.hrp
    etichette = list(a.stima.covarianza.index)
    d = h.distanze.to_numpy().copy()
    np.fill_diagonal(d, 0.0)
    fig = ff.create_dendrogram(d, labels=etichette, linkagefun=lambda _: h.collegamenti,
                               distfun=lambda x: x[np.triu_indices(len(x), 1)],
                               colorscale=[BLU_SCURO, ACQUA, VIOLA, ARANCIO, GIALLO, ROSSO, GRIGIO, BLU])
    fig.update_yaxes(title="Distanza di correlazione")
    fig.update_xaxes(range=[0, 10 * len(etichette)], tickangle=-35)
    fig.update_layout(showlegend=False)
    fig = _layout(fig, "Come HRP raggruppa gli asset (clustering gerarchico)", 470)
    fig.update_layout(margin={"b": 120})
    return fig


def grafico_correlazioni(a: Analisi) -> go.Figure:
    from .stime import correlazione

    ordine = a.hrp.ordine
    c = correlazione(a.stima.covarianza).loc[ordine, ordine]
    fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=ordine, y=ordine, zmin=-1, zmax=1,
                               colorscale=[[0, BLU_SCURO], [0.5, "#f7f7f5"], [1, ROSSO]],
                               text=np.round(c.to_numpy(), 2), texttemplate="%{text}",
                               hovertemplate="%{y} – %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Correlazioni nell'ordine trovato da HRP", 500)


def grafico_cvar(a: Analisi) -> go.Figure:
    fr = a.frontiera_cvar
    fig = go.Figure(go.Scatter(x=fr["CVaR"], y=fr["Rendimento"] * 12, mode="lines+markers",
                               name="Frontiera rendimento-CVaR", line={"color": VIOLA, "width": 2.5},
                               hovertemplate="CVaR %{x:.2%}<br>Rendimento %{y:.2%}<extra></extra>"))
    r = a.rendimenti
    from .cvar import cvar_empirico
    for nome, p in a.portafogli.items():
        s = r @ p.pesi.reindex(r.columns).fillna(0)
        fig.add_trace(go.Scatter(x=[cvar_empirico(s.to_numpy(), a.config.alfa)], y=[s.mean() * 12], mode="markers",
                                 name=nome, marker={"size": 13 if nome == PROPOSTO else 10,
                                                    "color": COLORI_STRATEGIE.get(nome, GRIGIO),
                                                    "symbol": "star" if nome == PROPOSTO else "circle"},
                                 hovertemplate=nome + "<br>CVaR %{x:.2%}<br>Rendimento %{y:.2%}<extra></extra>"))
    fig.update_xaxes(tickformat=".1%", title=f"CVaR {a.config.alfa:.0%} mensile (perdita media nei mesi peggiori)")
    fig.update_yaxes(tickformat=".1%", title="Rendimento medio annuo storico")
    return _layout(fig, "Rischio di coda: la frontiera a minimo CVaR", 480)


def grafico_operazioni(a: Analisi) -> go.Figure:
    op = a.operazioni["Operazione"]
    op = op[op.abs() > 0]
    fig = go.Figure(go.Bar(x=op.index, y=op, marker_color=[ACQUA if v > 0 else ROSSO for v in op],
                           text=[_eur(v) for v in op], textposition="outside",
                           hovertemplate="%{x}: %{y:,.0f} €<extra></extra>"))
    fig.update_yaxes(title="Acquisti (+) e vendite (−) in euro", tickformat=",.0f")
    return _etichette(_layout(fig, "Le operazioni per passare al portafoglio proposto", 440))


def grafico_backtest(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for nome, x in a.backtest.items():
        valore = 100 * (1 + x.rendimenti).cumprod()
        fig.add_trace(go.Scatter(x=valore.index, y=valore, name=nome, mode="lines",
                                 line={"color": COLORI_STRATEGIE.get(nome, GRIGIO),
                                       "width": 2.6 if nome == "Black-Litterman" else 1.6},
                                 hovertemplate=nome + "<br>%{x|%m/%Y}: %{y:.1f}<extra></extra>"))
    fig.update_yaxes(title="Valore (base 100)")
    return _layout(fig, "Backtest walk-forward, al netto dei costi", 480)


def grafico_drawdown(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for nome, x in a.backtest.items():
        dd = metriche.drawdown(x.rendimenti)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=nome, mode="lines",
                                 line={"color": COLORI_STRATEGIE.get(nome, GRIGIO), "width": 1.6},
                                 hovertemplate=nome + "<br>%{x|%m/%Y}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="Perdita dal massimo")
    return _layout(fig, "Perdite dal massimo nel backtest", 420)


def grafico_pesi_tempo(a: Analisi) -> go.Figure:
    nomi = [n for n in ("Media-varianza storica", "Black-Litterman") if n in a.backtest]
    fig = make_subplots(rows=1, cols=len(nomi), shared_yaxes=True, subplot_titles=nomi, horizontal_spacing=0.04)
    col = _colore_asset(a)
    for j, nome in enumerate(nomi, start=1):
        p = a.backtest[nome].pesi
        for x in p.columns:
            fig.add_trace(go.Scatter(x=p.index, y=p[x], name=x, stackgroup=f"g{j}", mode="lines",
                                     line={"width": 0.5, "color": col[x]}, legendgroup=x, showlegend=j == 1,
                                     hovertemplate=x + " %{x|%m/%Y}: %{y:.1%}<extra></extra>"), row=1, col=j)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig = _layout(fig, "Pesi scelti nel tempo: media-varianza storica e Black-Litterman", 470)
    fig.update_annotations(font={"size": 12, "color": TESTO_SECONDARIO})
    return fig


GRAFICI = {
    "rendimenti": grafico_rendimenti,
    "allocazioni": grafico_allocazioni,
    "frontiere": grafico_frontiere,
    "instabilita": grafico_instabilita,
    "dendrogramma": grafico_dendrogramma,
    "correlazioni": grafico_correlazioni,
    "cvar": grafico_cvar,
    "contributi": grafico_contributi,
    "operazioni": grafico_operazioni,
    "backtest": grafico_backtest,
    "drawdown": grafico_drawdown,
    "pesi_tempo": grafico_pesi_tempo,
}


def _disponibili(a: Analisi) -> dict:
    g = dict(GRAFICI)
    if not a.backtest:
        for k in ("backtest", "drawdown", "pesi_tempo"):
            g.pop(k)
    if not (a.operazioni["Operazione"].abs() > 0).any():
        g.pop("operazioni")
    return g


def salva_grafici(a: Analisi, cartella: str | Path, scala: float = 2.0) -> list[Path]:
    """Esporta i grafici in PNG (richiede ``kaleido`` e un browser Chrome/Chromium)."""
    out = Path(cartella)
    out.mkdir(parents=True, exist_ok=True)
    percorsi = []
    for nome, funzione in _disponibili(a).items():
        percorso = out / f"{nome}.png"
        funzione(a).write_image(percorso, width=900, scale=scala)
        percorsi.append(percorso)
    return percorsi


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float, decimali: int = 1) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.{decimali}%}".replace(".", ",").replace("-", "−")


def _num(x: float, decimali: int = 2) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.{decimali}f}".replace(".", ",").replace("-", "−")


def _eur(x: float) -> str:
    if pd.isna(x):
        return "–"
    testo = f"{abs(x):,.0f}".replace(",", ".")
    return f"−€ {testo}" if x < -0.5 else f"€ {testo}"


def _sintesi(a: Analisi) -> tuple[str, str]:
    att, pro = a.confronto.loc["Attuale"], a.confronto.loc[PROPOSTO]
    p = a.proposto
    testo = (f"Il portafoglio proposto combina l'equilibrio di mercato con {len(a.bl.viste)} view del consulente e "
             f"rispetta tutti i vincoli del mandato. Rispetto al portafoglio attuale il rendimento atteso passa da "
             f"{_pct(att['Rendimento atteso'], 2)} a {_pct(pro['Rendimento atteso'], 2)} e la volatilità attesa da "
             f"{_pct(att['Volatilità attesa'])} a {_pct(pro['Volatilità attesa'])}; per arrivarci si sposta il "
             f"{_pct(p.rotazione)} del patrimonio, con costi stimati di {_eur(p.costi * a.config.importo)}.")
    if not a.metriche_backtest.empty:
        m = a.metriche_backtest
        testo += (f" Nel backtest la media-varianza con le stime storiche ha ruotato il "
                  f"{_pct(m.loc['Media-varianza storica', 'Rotazione annua'], 0)} del portafoglio all'anno, "
                  f"Black-Litterman il {_pct(m.loc['Black-Litterman', 'Rotazione annua'], 0)}.")
    migliora = pro["Sharpe atteso"] >= att["Sharpe atteso"] - 1e-9
    return testo, "ok" if migliora else "attenzione"


def build_report(a: Analisi, output: str | Path, data_note: str = "", offline: bool = False) -> Path:
    """Scrive il report HTML. Con ``offline=True`` include plotly.js nel file (circa 4 MB)."""
    c = a.config
    sintesi, esito = _sintesi(a)
    config_plotly = {"displaylogo": False, "responsive": True}
    grafici = {nome: f(a).to_html(full_html=False, include_plotlyjs=False, config=config_plotly)
               for nome, f in _disponibili(a).items()}
    p, att = a.confronto.loc[PROPOSTO], a.confronto.loc["Attuale"]
    lv = f"{c.alfa:.0%}"
    kpi = [
        ("Rendimento atteso", f"{_pct(p['Rendimento atteso'], 2)}", f"attuale {_pct(att['Rendimento atteso'], 2)}"),
        ("Volatilità attesa", _pct(p["Volatilità attesa"]), f"attuale {_pct(att['Volatilità attesa'])}"),
        ("Sharpe atteso", _num(p["Sharpe atteso"]), f"attuale {_num(att['Sharpe atteso'])}"),
        (f"CVaR {lv} mensile", _eur(p[f"CVaR {lv} mensile storico"] * c.importo),
         f"attuale {_eur(att[f'CVaR {lv} mensile storico'] * c.importo)}"),
        ("Patrimonio da spostare", _pct(a.proposto.rotazione), _eur(a.proposto.rotazione * c.importo)),
        ("Costi stimati", _eur(a.proposto.costi * c.importo), f"{_num(c.costi * 1e4, 0)} pb per operazione"),
    ]
    asset = list(a.rendimenti.columns)
    allocazione = [{"nome": x, "att": _pct(r["Peso attuale"]), "pro": _pct(r["Peso proposto"]),
                    "var": _pct(r["Variazione"]), "eur": _eur(r["Controvalore proposto"]),
                    "op": _eur(r["Operazione"]) if r["Operazione"] else "–",
                    "segno": "pos" if r["Operazione"] > 0 else "neg" if r["Operazione"] < 0 else ""}
                   for x, r in a.operazioni.iterrows()]
    rendimenti = [{"nome": x, "storico": _pct(a.stima.media[x], 2), "eq": _pct(a.bl.equilibrio[x], 2),
                   "bl": _pct(a.bl.posteriore[x], 2), "vol": _pct(np.sqrt(a.stima.covarianza.loc[x, x])),
                   "mercato": _pct(c.pesi_mercato.get(x, 0.0))} for x in asset]
    viste = [{"descr": r["View"], "fiducia": _pct(r["Fiducia"], 0), "indicato": _pct(r["Indicato"], 2),
              "eq": _pct(r["Equilibrio"], 2), "post": _pct(r["A posteriori"], 2)}
             for _, r in a.bl.tabella_viste().iterrows()]
    vincoli = []
    v = c.vincoli
    vincoli.append(f"peso di ciascun asset tra {_pct(v.minimo if not isinstance(v.minimo, dict) else 0, 0)} e "
                   f"{_pct(v.massimo if not isinstance(v.massimo, dict) else 1, 0)}")
    for nome, (_, lo, hi) in v.gruppi.items():
        vincoli.append(f"{nome.lower()} tra {_pct(lo, 0)} e {_pct(hi, 0)}")
    if c.rotazione_massima is not None:
        vincoli.append(f"al massimo il {_pct(c.rotazione_massima, 0)} del patrimonio spostato")
    colonne = ["Rendimento atteso", "Volatilità attesa", "Sharpe atteso", "Equivalente certo",
               f"CVaR {lv} mensile storico",
               "Massima perdita storica", "Asset effettivi", "Rapporto di diversificazione", "Peso massimo"]
    fmt = {"Sharpe atteso": _num, "Asset effettivi": lambda x: _num(x, 1), "Rapporto di diversificazione": _num,
           "Rendimento atteso": lambda x: _pct(x, 2), "Equivalente certo": lambda x: _pct(x, 2)}
    confronto = [{"nome": n, "forte": n == PROPOSTO,
                  "valori": [fmt.get(k, _pct)(a.confronto.loc[n, k]) for k in colonne]} for n in a.confronto.index]
    m = a.metriche_backtest
    colonne_bt = ["Rendimento annuo", "Volatilità", "Sharpe", "Massima perdita", "CVaR 95% mensile",
                  "Rotazione annua", "Costi annui", "Instabilità dei pesi"]
    fmt_bt = {"Sharpe": _num, "Rotazione annua": lambda x: _pct(x, 0), "Costi annui": lambda x: _pct(x, 2)}
    backtest = [{"nome": n, "valori": [fmt_bt.get(k, _pct)(m.loc[n, k]) for k in colonne_bt]} for n in m.index]
    if len(m):
        inizio_bt = next(iter(a.backtest.values())).rendimenti.index[0]
        migliore = m["Sharpe"].idxmax()
    else:
        inizio_bt, migliore = None, ""
    disp = a.ricampionamento.dispersione
    azionario = [x for x in c.vincoli.gruppi.get("Azionario", ([],))[0] if x in asset]
    quota_azioni = None
    if azionario:
        contr = metriche.contributi_rischio(a.proposto.pesi, a.stima.covarianza)
        quota_azioni = (_pct(a.proposto.pesi[azionario].sum(), 0), _pct(contr[azionario].sum(), 0))

    env = Environment(loader=PackageLoader("robust_allocation", "templates"), autoescape=select_autoescape())
    html = env.get_template("report.html.j2").render(
        cliente=c.cliente, importo=_eur(c.importo), oggi=date.today().strftime("%d/%m/%Y"),
        periodo=f"{a.rendimenti.index[0]:%m/%Y} al {a.rendimenti.index[-1]:%m/%Y}", mesi=len(a.rendimenti),
        sintesi=sintesi, esito=esito, kpi=kpi, allocazione=allocazione, rendimenti=rendimenti, viste=viste,
        vincoli=vincoli, confronto=confronto, colonne=colonne, backtest=backtest, colonne_bt=colonne_bt,
        grafici=grafici, nota=data_note, lv=lv, rf=_pct(c.risk_free, 2), avversione=_num(c.avversione, 1),
        tau=_num(c.tau, 2), costi_pb=_num(c.costi * 1e4, 0), shrink=_pct(a.stima.intensita, 0),
        inizio_bt=f"{inizio_bt:%m/%Y}" if inizio_bt is not None else "", migliore=migliore,
        n_sim=len(a.ricampionamento.simulazioni), disp_max=_pct(disp.max(), 0), disp_asset=disp.idxmax(),
        quota_azioni=quota_azioni, hrp_max=a.hrp.pesi.idxmax(), hrp_max_peso=_pct(a.hrp.pesi.max(), 0),
        cvar_var=_pct(a.cvar.var), cvar_cvar=_pct(a.cvar.cvar),
        plotly_inline=get_plotlyjs() if offline else None, plotly_cdn=PLOTLY_CDN,
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
