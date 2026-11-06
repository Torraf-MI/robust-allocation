import numpy as np
import pandas as pd
import pytest

from robust_allocation import dati


@pytest.fixture(scope="session")
def rendimenti_demo():
    return dati.rendimenti(dati.prezzi_dimostrativi())


@pytest.fixture
def universo_piccolo():
    """Quattro asset con media e covarianza note."""
    asset = ["A", "B", "C", "D"]
    vol = np.array([0.20, 0.15, 0.08, 0.05])
    corr = np.array([[1.0, 0.6, 0.2, 0.0], [0.6, 1.0, 0.3, 0.1], [0.2, 0.3, 1.0, 0.4], [0.0, 0.1, 0.4, 1.0]])
    cov = pd.DataFrame(corr * np.outer(vol, vol), index=asset, columns=asset)
    mu = pd.Series([0.08, 0.07, 0.04, 0.03], index=asset)
    return mu, cov
