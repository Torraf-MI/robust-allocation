import numpy as np
import pandas as pd
import pytest

from robust_allocation.ottimizzatore import Ottimizzatore, Vincoli, soluzione_analitica


def test_senza_vincoli_coincide_con_la_formula_chiusa(universo_piccolo):
    mu, cov = universo_piccolo
    p = Ottimizzatore(mu, cov, Vincoli(minimo=-5, massimo=5)).media_varianza(avversione=3.0)
    np.testing.assert_allclose(p.pesi, soluzione_analitica(mu.to_numpy(), cov.to_numpy(), 3.0), atol=1e-6)


def test_rispetta_limiti_e_gruppi(universo_piccolo):
    mu, cov = universo_piccolo
    v = Vincoli(minimo={"D": 0.10}, massimo=0.40, gruppi={"Rischiosi": (["A", "B"], 0.20, 0.50)})
    p = Ottimizzatore(mu, cov, v).media_varianza(avversione=1.0)
    w = p.pesi
    assert w.sum() == pytest.approx(1.0)
    assert (w <= 0.40 + 1e-8).all() and (w >= -1e-10).all()
    assert w["D"] >= 0.10 - 1e-8
    assert 0.20 - 1e-8 <= w[["A", "B"]].sum() <= 0.50 + 1e-8


def test_limite_di_rotazione(universo_piccolo):
    mu, cov = universo_piccolo
    attuali = pd.Series([0.0, 0.0, 0.5, 0.5], index=mu.index)
    libero = Ottimizzatore(mu, cov, Vincoli(pesi_attuali=attuali)).media_varianza(avversione=2.0)
    limitato = Ottimizzatore(mu, cov, Vincoli(pesi_attuali=attuali, rotazione_massima=0.10)).media_varianza(2.0)
    assert libero.rotazione > 0.10
    assert limitato.rotazione == pytest.approx(0.10, abs=1e-6)


def test_costi_alti_bloccano_le_operazioni(universo_piccolo):
    mu, cov = universo_piccolo
    attuali = pd.Series(0.25, index=mu.index)
    p = Ottimizzatore(mu, cov, Vincoli(pesi_attuali=attuali, costi=1.0)).media_varianza(avversione=2.0)
    np.testing.assert_allclose(p.pesi, attuali, atol=1e-6)
    assert p.costi == pytest.approx(0.0, abs=1e-6)


def test_costi_riducono_la_rotazione(universo_piccolo):
    mu, cov = universo_piccolo
    attuali = pd.Series(0.25, index=mu.index)
    zero = Ottimizzatore(mu, cov, Vincoli(pesi_attuali=attuali)).media_varianza(avversione=2.0)
    senza = Ottimizzatore(mu, cov).media_varianza(avversione=2.0)
    con = Ottimizzatore(mu, cov, Vincoli(pesi_attuali=attuali, costi=0.01)).media_varianza(avversione=2.0)
    np.testing.assert_allclose(zero.pesi, senza.pesi, atol=1e-5)
    assert con.rotazione < zero.rotazione


def test_minima_varianza_con_rendimento_minimo(universo_piccolo):
    mu, cov = universo_piccolo
    opt = Ottimizzatore(mu, cov)
    base = opt.minima_varianza()
    vincolato = opt.minima_varianza(rendimento_minimo=0.06)
    assert vincolato.rendimento >= 0.06 - 1e-8
    assert vincolato.volatilita > base.volatilita


def test_frontiera_crescente(universo_piccolo):
    mu, cov = universo_piccolo
    fr = Ottimizzatore(mu, cov, Vincoli(massimo=0.6)).frontiera(12)
    assert len(fr) >= 10
    assert np.all(np.diff(fr["Rendimento"]) > 0)
    assert np.all(np.diff(fr["Volatilità"]) > -1e-8)
