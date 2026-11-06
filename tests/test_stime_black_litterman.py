import numpy as np
import pandas as pd
import pytest

from robust_allocation import black_litterman as bl
from robust_allocation.stime import covarianza_ledoit_wolf, stima

MERCATO = pd.Series([0.4, 0.3, 0.2, 0.1], index=["A", "B", "C", "D"])


def test_ledoit_wolf_intensita_e_definita_positiva(rendimenti_demo):
    cov, k = covarianza_ledoit_wolf(rendimenti_demo)
    assert 0 <= k <= 1
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    campionaria = np.cov(rendimenti_demo.to_numpy(), rowvar=False, ddof=0)
    # Le varianze non vengono toccate: lo shrinkage agisce solo sulle correlazioni.
    np.testing.assert_allclose(np.diag(cov), np.diag(campionaria))


def test_ledoit_wolf_si_riduce_con_tanti_dati():
    # Correlazioni molto diverse tra loro: il bersaglio a correlazione costante è sbagliato,
    # quindi con tanti dati lo shrinkage deve quasi sparire.
    rng = np.random.default_rng(1)
    corr = np.array([[1.0, 0.8, 0.0, -0.3], [0.8, 1.0, 0.1, -0.2], [0.0, 0.1, 1.0, 0.6], [-0.3, -0.2, 0.6, 1.0]])
    chol = np.linalg.cholesky(corr)
    pochi = rng.standard_normal((30, 4)) @ chol.T
    tanti = rng.standard_normal((20_000, 4)) @ chol.T
    assert covarianza_ledoit_wolf(tanti)[1] < covarianza_ledoit_wolf(pochi)[1]
    assert covarianza_ledoit_wolf(tanti)[1] < 0.01


def test_ledoit_wolf_bersaglio_giusto_shrinkage_forte():
    rng = np.random.default_rng(2)
    _, k = covarianza_ledoit_wolf(rng.standard_normal((60, 8)))
    assert k > 0.5


def test_stima_annualizza(rendimenti_demo):
    e = stima(rendimenti_demo, shrinkage=False)
    np.testing.assert_allclose(e.media, rendimenti_demo.mean() * 12)
    np.testing.assert_allclose(e.covarianza, np.cov(rendimenti_demo.to_numpy(), rowvar=False, ddof=0) * 12)


def test_senza_view_restituisce_il_mercato(universo_piccolo):
    _, cov = universo_piccolo
    r = bl.modello(cov, MERCATO, [], avversione=3.0, risk_free=0.02)
    pd.testing.assert_series_equal(r.posteriore, r.equilibrio, check_names=False)
    np.testing.assert_allclose(r.pesi_equilibrio, MERCATO)
    np.testing.assert_allclose(r.equilibrio - 0.02, 3.0 * cov.to_numpy() @ MERCATO.to_numpy())


def test_view_certa_viene_rispettata(universo_piccolo):
    _, cov = universo_piccolo
    viste = [bl.Vista("relativa", ["A"], 0.03, 1.0, ["B"]), bl.Vista("assoluta", ["C"], 0.06, 1.0)]
    r = bl.modello(cov, MERCATO, viste, risk_free=0.02)
    mu = r.posteriore.to_numpy() - 0.02
    assert r.p[0] @ mu == pytest.approx(0.03)
    assert r.p[1] @ mu + 0.02 == pytest.approx(0.06)
    t = r.tabella_viste()
    np.testing.assert_allclose(t["A posteriori"], t["Indicato"])


@pytest.mark.parametrize("fiducia", [0.1, 0.3, 0.7, 0.95])
def test_fiducia_di_idzorek_sposta_i_pesi_della_quota_indicata(universo_piccolo, fiducia):
    _, cov = universo_piccolo
    vista = bl.Vista("relativa", ["C"], 0.04, fiducia, ["A"])
    piena = bl.modello(cov, MERCATO, [bl.Vista("relativa", ["C"], 0.04, 1.0, ["A"])])
    parziale = bl.modello(cov, MERCATO, [vista])
    inclinazione_piena = piena.pesi_bl - MERCATO
    inclinazione = parziale.pesi_bl - MERCATO
    np.testing.assert_allclose(inclinazione, fiducia * inclinazione_piena, atol=1e-10)


def test_fiducia_nulla_lascia_l_equilibrio(universo_piccolo):
    _, cov = universo_piccolo
    r = bl.modello(cov, MERCATO, [bl.Vista("assoluta", ["A"], 0.30, 0.0)])
    np.testing.assert_allclose(r.posteriore, r.equilibrio, atol=1e-6)


def test_he_litterman_coincide_con_fiducia_del_50_percento(universo_piccolo):
    _, cov = universo_piccolo
    viste = [bl.Vista("assoluta", ["B"], 0.09, 0.5), bl.Vista("relativa", ["D"], 0.01, 0.5, ["C"])]
    a = bl.modello(cov, MERCATO, viste, metodo_omega="idzorek")
    b = bl.modello(cov, MERCATO, viste, metodo_omega="he-litterman")
    np.testing.assert_allclose(a.posteriore, b.posteriore)


def test_covarianza_posteriore_tra_prior_e_prior_piu_tau(universo_piccolo):
    _, cov = universo_piccolo
    s = cov.to_numpy()
    r = bl.modello(cov, MERCATO, [bl.Vista("assoluta", ["A"], 0.1, 0.5)], tau=0.05)
    post = r.covarianza.to_numpy()
    assert np.all(np.linalg.eigvalsh(post - s) >= -1e-12)
    assert np.all(np.linalg.eigvalsh(s * 1.05 - post) >= -1e-12)


def test_view_non_valide():
    with pytest.raises(ValueError):
        bl.Vista("boh", ["A"], 0.01)
    with pytest.raises(ValueError):
        bl.Vista("relativa", ["A"], 0.01)
    with pytest.raises(KeyError):
        bl.matrici_viste([bl.Vista("assoluta", ["Z"], 0.01)], ["A", "B"])


def test_vista_da_dict():
    v = bl.Vista.da_dict({"tipo": "assoluta", "asset": "Oro", "rendimento": 0.05, "fiducia": 0.25})
    assert v.lunghi == ["Oro"] and v.fiducia == 0.25 and "Oro" in v.descrizione


def test_viste_momentum_usano_solo_il_passato():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    r = pd.DataFrame({"A": 0.02, "B": 0.0, "C": -0.01}, index=idx)
    v = bl.viste_momentum(r, mesi=12)[0]
    assert v.lunghi == ["A"] and v.corti == ["C"]
    assert bl.viste_momentum(r.iloc[:6], mesi=12) == []
