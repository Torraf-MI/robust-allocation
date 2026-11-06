"""Modello di Black-Litterman: equilibrio di mercato + view del consulente con il loro grado di fiducia.

Tutti i rendimenti sono **in eccesso sul tasso privo di rischio** dentro il modello; le view assolute vengono
espresse in rendimento totale e convertite. Riferimenti: Black e Litterman (1992), He e Litterman (1999),
Idzorek (2005) per la fiducia in percentuale, Walters (2011) per la sua forma chiusa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AVVERSIONE_DEFAULT = 2.5
TAU_DEFAULT = 0.05
_FIDUCIA_MINIMA = 1e-6


@dataclass
class Vista:
    """Una view del consulente.

    - ``assoluta``: «l'asset ``lunghi[0]`` renderà ``rendimento`` all'anno»;
    - ``relativa``: «i ``lunghi`` faranno ``rendimento`` all'anno meglio dei ``corti``» (pesi uguali per gamba).

    ``fiducia`` è tra 0 e 1, nel senso di Idzorek: con 0,5 il portafoglio si sposta di metà della distanza tra
    l'equilibrio e il portafoglio che si avrebbe credendo alla view con certezza.
    """

    tipo: str
    lunghi: list[str]
    rendimento: float
    fiducia: float = 0.5
    corti: list[str] = field(default_factory=list)
    descrizione: str = ""

    def __post_init__(self) -> None:
        if self.tipo not in ("assoluta", "relativa"):
            raise ValueError(f"Tipo di view sconosciuto: {self.tipo}")
        if self.tipo == "relativa" and not self.corti:
            raise ValueError("Una view relativa ha bisogno di almeno un asset in 'corti'")
        if not 0 <= self.fiducia <= 1:
            raise ValueError("La fiducia deve essere tra 0 e 1")
        if not self.descrizione:
            r = f"{self.rendimento:.1%}".replace(".", ",")
            if self.tipo == "assoluta":
                self.descrizione = f"{self.lunghi[0]} renderà il {r} all'anno"
            else:
                self.descrizione = f"{', '.join(self.lunghi)} meglio di {', '.join(self.corti)} di {r} all'anno"

    @classmethod
    def da_dict(cls, d: dict) -> Vista:
        lunghi = d.get("lunghi") or [d["asset"]]
        return cls(d["tipo"], list(lunghi), float(d["rendimento"]), float(d.get("fiducia", 0.5)),
                   list(d.get("corti", [])), d.get("descrizione", ""))


def avversione_implicita(premio_mercato: float, volatilita_mercato: float) -> float:
    """Avversione al rischio coerente con il premio atteso del mercato: δ = (E[Rm] − rf) / σm²."""
    return premio_mercato / volatilita_mercato**2


def rendimenti_equilibrio(cov: pd.DataFrame, pesi_mercato: pd.Series, avversione: float) -> pd.Series:
    """Rendimenti in eccesso impliciti: quelli per cui il portafoglio di mercato è ottimo (π = δ Σ w)."""
    w = pesi_mercato.reindex(cov.index).fillna(0.0)
    return (avversione * cov @ w).rename("Equilibrio")


def matrici_viste(viste: list[Vista], asset: list[str], risk_free: float = 0.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matrice P (k × n), vettore Q (in eccesso) e fiducie delle view."""
    idx = {a: i for i, a in enumerate(asset)}
    p = np.zeros((len(viste), len(asset)))
    q = np.zeros(len(viste))
    for k, v in enumerate(viste):
        for a in v.lunghi + v.corti:
            if a not in idx:
                raise KeyError(f"Asset della view non presente nell'universo: {a}")
        for a in v.lunghi:
            p[k, idx[a]] = 1 / len(v.lunghi)
        for a in v.corti:
            p[k, idx[a]] = -1 / len(v.corti)
        q[k] = v.rendimento - (risk_free if v.tipo == "assoluta" else 0.0)
    return p, q, np.array([v.fiducia for v in viste])


def omega_he_litterman(p: np.ndarray, cov: np.ndarray, tau: float) -> np.ndarray:
    """Incertezza delle view proporzionale alla varianza dei portafogli delle view: Ω = diag(P τΣ P')."""
    return np.diag(np.diag(p @ (tau * cov) @ p.T))


def omega_idzorek(p: np.ndarray, cov: np.ndarray, tau: float, fiducia: np.ndarray) -> np.ndarray:
    """Ω che realizza la fiducia percentuale di Idzorek, in forma chiusa: ωₖ = (1 − cₖ)/cₖ · pₖ τΣ pₖ'."""
    c = np.clip(np.asarray(fiducia, dtype=float), _FIDUCIA_MINIMA, 1.0)
    var_viste = np.diag(p @ (tau * cov) @ p.T)
    return np.diag((1 - c) / c * var_viste)


def posteriore(pi: np.ndarray, cov: np.ndarray, p: np.ndarray, q: np.ndarray, omega: np.ndarray,
               tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Media e covarianza a posteriori.

    Usa la forma che non richiede di invertire Ω, così funziona anche con view certe (Ω = 0):
    μ = π + τΣP'(PτΣP' + Ω)⁻¹(Q − Pπ),  Σ_post = Σ + τΣ − τΣP'(PτΣP' + Ω)⁻¹PτΣ.
    """
    ts = tau * cov
    if len(q) == 0:
        return pi.copy(), cov + ts
    a = p @ ts @ p.T + omega
    k = ts @ p.T @ np.linalg.solve(a, np.eye(len(q)))
    mu = pi + k @ (q - p @ pi)
    sigma = cov + ts - k @ p @ ts
    return mu, (sigma + sigma.T) / 2


def pesi_senza_vincoli(mu: np.ndarray, cov: np.ndarray, avversione: float) -> np.ndarray:
    """Pesi ottimi senza vincoli: w = (δΣ)⁻¹ μ (con μ in eccesso)."""
    return np.linalg.solve(avversione * cov, mu)


@dataclass
class RisultatoBL:
    asset: list[str]
    equilibrio: pd.Series  # rendimenti totali di equilibrio
    posteriore: pd.Series  # rendimenti totali a posteriori
    covarianza: pd.DataFrame  # covarianza a posteriori
    pesi_equilibrio: pd.Series
    pesi_bl: pd.Series  # pesi senza vincoli con le view
    viste: list[Vista]
    p: np.ndarray
    q: np.ndarray
    omega: np.ndarray
    avversione: float
    tau: float
    risk_free: float

    def tabella_viste(self) -> pd.DataFrame:
        """Per ogni view: rendimento indicato, implicito nell'equilibrio e risultante a posteriori."""
        pi = self.equilibrio.to_numpy() - self.risk_free
        mu = self.posteriore.to_numpy() - self.risk_free
        righe = []
        for k, v in enumerate(self.viste):
            agg = self.risk_free if v.tipo == "assoluta" else 0.0
            righe.append({
                "View": v.descrizione,
                "Fiducia": v.fiducia,
                "Indicato": v.rendimento,
                "Equilibrio": self.p[k] @ pi + agg,
                "A posteriori": self.p[k] @ mu + agg,
            })
        return pd.DataFrame(righe)


def modello(cov: pd.DataFrame, pesi_mercato: pd.Series, viste: list[Vista] | None = None,
                    avversione: float = AVVERSIONE_DEFAULT, tau: float = TAU_DEFAULT, risk_free: float = 0.0,
                    metodo_omega: str = "idzorek") -> RisultatoBL:
    """Combina l'equilibrio di mercato con le view; restituisce rendimenti totali e covarianza a posteriori."""
    viste = viste or []
    asset = list(cov.index)
    s = cov.to_numpy()
    pi = rendimenti_equilibrio(cov, pesi_mercato, avversione).to_numpy()
    p, q, fiducia = matrici_viste(viste, asset, risk_free)
    if metodo_omega == "idzorek":
        omega = omega_idzorek(p, s, tau, fiducia)
    elif metodo_omega == "he-litterman":
        omega = omega_he_litterman(p, s, tau)
    else:
        raise ValueError(f"Metodo per Ω sconosciuto: {metodo_omega}")
    mu, sigma = posteriore(pi, s, p, q, omega, tau)
    return RisultatoBL(
        asset=asset,
        equilibrio=pd.Series(pi + risk_free, index=asset, name="Equilibrio"),
        posteriore=pd.Series(mu + risk_free, index=asset, name="Black-Litterman"),
        covarianza=pd.DataFrame(sigma, index=asset, columns=asset),
        pesi_equilibrio=pd.Series(pesi_senza_vincoli(pi, s, avversione), index=asset),
        pesi_bl=pd.Series(pesi_senza_vincoli(mu, s, avversione), index=asset),
        viste=viste, p=p, q=q, omega=omega, avversione=avversione, tau=tau, risk_free=risk_free,
    )


def viste_momentum(r: pd.DataFrame, mesi: int = 12, fiducia: float = 0.25, scarto: float = 0.02) -> list[Vista]:
    """View quantitative senza informazioni future: il terzo più forte degli ultimi ``mesi`` batte il più debole.

    Utili nel backtest, dove le view del consulente non sono disponibili data per data.
    """
    if len(r) < mesi:
        return []
    mom = (1 + r.iloc[-mesi:]).prod() - 1
    ordine = mom.sort_values(ascending=False).index.tolist()
    terzo = max(1, len(ordine) // 3)
    return [Vista("relativa", ordine[:terzo], scarto, fiducia, ordine[-terzo:],
                  descrizione=f"Momentum a {mesi} mesi")]
