import json

import numpy as np
import pandas as pd
import pytest

from robust_allocation import backtest as bt
from robust_allocation import configurazione, dati
from robust_allocation.analisi import PROPOSTO, analizza
from robust_allocation.cli import main
from robust_allocation.report import build_report


def _rendimenti(n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    return pd.DataFrame(rng.normal(0.005, 0.03, (n, 2)), index=idx, columns=["X", "Y"])


def test_motore_coincide_con_il_calcolo_a_mano():
    r = _rendimenti()
    fissi = pd.Series({"X": 0.6, "Y": 0.4})
    ris = bt.esegui(r, lambda storia, attuali: fissi, "Fissi", minimo_storia=12, ogni=3, costi=0.001)
    w = fissi.copy()
    attesi = []
    for k, t in enumerate(range(12, len(r))):
        costo = 0.0
        if k == 0:
            costo = 0.001
        elif k % 3 == 0:
            costo = 0.001 * (fissi - w).abs().sum()
            w = fissi.copy()
        lordo = float(w @ r.iloc[t])
        attesi.append(lordo - costo)
        w = w * (1 + r.iloc[t]) / (1 + lordo)
    np.testing.assert_allclose(ris.rendimenti.to_numpy(), attesi)


def test_nessuna_informazione_futura():
    r = _rendimenti(40)
    viste = []

    def strategia(storia, attuali):
        viste.append(storia.index[-1])
        return pd.Series(storia.mean().clip(lower=1e-6))

    bt.esegui(r, strategia, "s", minimo_storia=12, ogni=3)
    date_decisione = bt.esegui(r, lambda s, a: pd.Series(1.0, index=s.columns), "u", 12, 3).pesi.index
    for vista, decisione in zip(viste, date_decisione):
        assert vista < decisione

    alterato = r.copy()
    alterato.iloc[25:] = 0.5
    a = bt.esegui(r, lambda s, _: s.mean().clip(lower=1e-6), "a", 12, 3).pesi
    b = bt.esegui(alterato, lambda s, _: s.mean().clip(lower=1e-6), "b", 12, 3).pesi
    pd.testing.assert_frame_equal(a.loc[: r.index[25]], b.loc[: r.index[25]])


def test_costi_riducono_il_rendimento():
    r = _rendimenti(40)
    alterna = iter(range(1000))

    def strategia(storia, _):
        return pd.Series({"X": 0.9, "Y": 0.1}) if next(alterna) % 2 else pd.Series({"X": 0.1, "Y": 0.9})

    senza = bt.esegui(r, strategia, "a", 12, 3, costi=0.0)
    alterna = iter(range(1000))
    con = bt.esegui(r, strategia, "b", 12, 3, costi=0.01)
    assert (con.rendimenti <= senza.rendimenti + 1e-12).all()
    assert con.rotazione_annua > 0


@pytest.fixture(scope="module")
def analisi_demo():
    return analizza(dati.prezzi_dimostrativi(), configurazione.esempio(), backtest=False, simulazioni=40)


def test_portafoglio_proposto_rispetta_il_mandato(analisi_demo):
    cfg = configurazione.esempio()
    w = analisi_demo.proposto.pesi
    assert w.sum() == pytest.approx(1.0)
    assert (w <= cfg.vincoli.massimo + 1e-8).all()
    for membri, lo, hi in cfg.vincoli.gruppi.values():
        assert lo - 1e-8 <= w[membri].sum() <= hi + 1e-8
    assert analisi_demo.proposto.rotazione <= cfg.rotazione_massima + 1e-8
    op = analisi_demo.operazioni
    assert op["Operazione"].sum() == pytest.approx(0.0, abs=5)


def test_report_e_tabelle(analisi_demo, tmp_path):
    assert set(analisi_demo.confronto.index) >= {"Attuale", "Mercato", "HRP", "Minimo CVaR", PROPOSTO}
    path = build_report(analisi_demo, tmp_path / "r.html", data_note="Prova")
    html = path.read_text(encoding="utf-8")
    assert "Black-Litterman" in html and "Prova" in html and "Famiglia" not in html


def test_backtest_completo_breve():
    prezzi = dati.prezzi_dimostrativi().iloc[:100]
    a = analizza(prezzi, configurazione.esempio(), simulazioni=20, simulazioni_backtest=5)
    assert set(a.backtest) >= {"Black-Litterman", "HRP", "Minimo CVaR"}
    assert (a.metriche_backtest["Rotazione annua"] >= 0).all()


def test_riga_di_comando_e_configurazione(tmp_path):
    cfg = json.loads(open("esempi/cliente.json", encoding="utf-8").read())
    cfg["prezzi"] = str((tmp_path / "p.csv").resolve())
    dati.prezzi_dimostrativi().iloc[:90].to_csv(tmp_path / "p.csv")
    (tmp_path / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "report.html"
    assert main(["--config", str(tmp_path / "c.json"), "--output", str(out), "--simulazioni", "20",
                 "--senza-backtest"]) == 0
    assert "Famiglia Bianchi" in out.read_text(encoding="utf-8")
