import numpy as np
import pandas as pd
import pytest

from robust_allocation.cvar import cvar_empirico, frontiera_cvar, minimo_cvar
from robust_allocation.hrp import hrp
from robust_allocation.metriche import numero_effettivo
from robust_allocation.ottimizzatore import Ottimizzatore, Vincoli
from robust_allocation.ricampionamento import ricampiona


def test_hrp_con_covarianza_diagonale_e_inverso_della_varianza():
    var = np.array([0.04, 0.01, 0.09, 0.0025, 0.0225])
    nomi = list("ABCDE")
    cov = pd.DataFrame(np.diag(var), index=nomi, columns=nomi)
    # Con correlazioni nulle il clustering è indifferente, il risultato no.
    w = hrp(cov + 1e-12).pesi
    attesi = (1 / var) / (1 / var).sum()
    np.testing.assert_allclose(w, attesi, rtol=1e-6)


def test_hrp_divide_tra_due_blocchi_uguali():
    blocco = np.array([[0.04, 0.03], [0.03, 0.04]])
    cov = np.zeros((4, 4))
    cov[:2, :2] = blocco
    cov[2:, 2:] = blocco
    nomi = ["A1", "A2", "B1", "B2"]
    ris = hrp(pd.DataFrame(cov, index=nomi, columns=nomi))
    assert ris.pesi[["A1", "A2"]].sum() == pytest.approx(0.5)
    assert set(ris.ordine[:2]) in ({"A1", "A2"}, {"B1", "B2"})


def test_hrp_pesi_positivi(rendimenti_demo):
    from robust_allocation.stime import stima

    w = hrp(stima(rendimenti_demo).covarianza).pesi
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_cvar_empirico():
    r = np.arange(-10, 10) / 100  # 20 osservazioni, perdita peggiore 10%
    assert cvar_empirico(r, 0.90) == pytest.approx(0.095)  # media di 10% e 9%
    assert cvar_empirico(r, 0.95) == pytest.approx(0.10)


def test_minimo_cvar_ottimo_coincide_con_il_cvar_del_portafoglio():
    rng = np.random.default_rng(3)
    scenari = pd.DataFrame(rng.standard_t(4, (1000, 4)) * [0.05, 0.04, 0.02, 0.01] + 0.005, columns=list("ABCD"))
    ris = minimo_cvar(scenari, 0.95)
    portafoglio = scenari.to_numpy() @ ris.pesi.to_numpy()
    assert ris.cvar == pytest.approx(cvar_empirico(portafoglio, 0.95), rel=1e-6)
    for _ in range(50):
        w = rng.dirichlet(np.ones(4))
        assert ris.cvar <= cvar_empirico(scenari.to_numpy() @ w, 0.95) + 1e-9


def test_minimo_cvar_rispetta_i_vincoli(rendimenti_demo):
    gruppi = {"Azionario": (["Azionario USA", "Azionario Europa"], 0.3, 0.5)}
    ris = minimo_cvar(rendimenti_demo, 0.95, Vincoli(massimo=0.25, gruppi=gruppi))
    assert (ris.pesi <= 0.25 + 1e-8).all()
    assert 0.3 - 1e-8 <= ris.pesi[["Azionario USA", "Azionario Europa"]].sum() <= 0.5 + 1e-8


def test_frontiera_cvar_crescente(rendimenti_demo):
    fr = frontiera_cvar(rendimenti_demo, 0.95, Vincoli(massimo=0.4), punti=8)
    assert np.all(np.diff(fr["CVaR"]) > -1e-10)
    assert np.all(np.diff(fr["Rendimento"]) > 0)


def test_ricampionamento_piu_diversificato(universo_piccolo):
    mu, cov = universo_piccolo
    v = Vincoli(massimo=0.7)
    classico = Ottimizzatore(mu, cov, v).media_varianza(2.0).pesi
    ris = ricampiona(mu, cov, osservazioni=60, avversione=2.0, vincoli=v, simulazioni=100, seed=1)
    assert ris.pesi.sum() == pytest.approx(1.0)
    assert (ris.pesi <= 0.7 + 1e-8).all()
    assert len(ris.simulazioni) == 100
    assert numero_effettivo(ris.pesi) > numero_effettivo(classico)
    assert (ris.dispersione > 0).all()
