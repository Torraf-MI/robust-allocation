"""Riga di comando: portafoglio proposto con Black-Litterman, confronto dei metodi, backtest e report.

Esempi::

    robust-allocation --demo --cliente "Famiglia Bianchi" --importo 750000
    robust-allocation --config esempi/cliente.json
    robust-allocation --demo --senza-backtest --offline
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import configurazione, dati
from .analisi import PROPOSTO, analizza
from .report import _eur, _pct, build_report, salva_grafici


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="robust-allocation", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="File JSON con prezzi, mandato, view e portafoglio attuale")
    src.add_argument("--demo", action="store_true", help="Usa dati, view e portafoglio dimostrativi")
    p.add_argument("--cliente", help="Nome del cliente nel report")
    p.add_argument("--importo", type=float, help="Patrimonio in euro")
    p.add_argument("--senza-backtest", action="store_true", help="Salta il backtest walk-forward (più veloce)")
    p.add_argument("--simulazioni", type=int, default=300, help="Simulazioni del ricampionamento (default 300)")
    p.add_argument("--seed", type=int, default=0, help="Seme dei numeri casuali")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (funziona senza rete)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    nota = ""
    if args.demo:
        prezzi, cfg = dati.prezzi_dimostrativi(), configurazione.esempio()
        nota = ("Analisi svolta su dati simulati, non su indici reali: le crisi ricalcano per date e ordine di "
                "grandezza episodi storici, ma i numeri non sono quelli di alcuno strumento.")
    else:
        prezzi, cfg = configurazione.carica(args.config)
    if args.cliente:
        cfg.cliente = args.cliente
    if args.importo:
        cfg.importo = args.importo

    a = analizza(prezzi, cfg, backtest=not args.senza_backtest, simulazioni=args.simulazioni, seed=args.seed)
    path = build_report(a, args.output, data_note=nota, offline=args.offline)
    if args.grafici:
        salva_grafici(a, args.grafici)

    pd.set_option("display.width", 180)
    print("Pesi per metodo")
    print(pd.DataFrame({k: p.pesi for k, p in a.portafogli.items()}).map(lambda x: _pct(x)).to_string())
    print("\nConfronto")
    print(a.confronto[["Rendimento atteso", "Volatilità attesa", "Sharpe atteso", "Asset effettivi"]].round(3)
          .to_string())
    p = a.portafogli[PROPOSTO]
    print(f"\nPortafoglio proposto: rotazione {_pct(p.rotazione)}, costi {_eur(p.costi * cfg.importo)}")
    if not a.metriche_backtest.empty:
        print("\nBacktest")
        print(a.metriche_backtest[["Rendimento annuo", "Volatilità", "Sharpe", "Massima perdita", "Rotazione annua"]]
              .round(3).to_string())
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
