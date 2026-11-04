"""Project-wide constants: universe, study period, sectors, corporate actions, paths."""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

RAW_PRICES = DATA_DIR / "prices_raw.csv"
CLEAN_PRICES = DATA_DIR / "prices_clean.csv"
CORRELATION_MATRIX = DATA_DIR / "correlation_matrix.csv"
DISTANCE_MATRIX = DATA_DIR / "distance_matrix.csv"
MST_EDGES = DATA_DIR / "mst_edges.csv"
NODE_METRICS = DATA_DIR / "mst_node_metrics.csv"

# ── Study period ─────────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE = "2026-06-01"
BOARD = "TQBR"  # main board for Russian shares (T+1)

# ── Universe: 44 liquid MOEX shares grouped by sector ────────────────────
# Nine sectors; the categorical palette has eight slots, so the smallest
# group (Transport) is drawn in neutral gray (see plotting.SECTOR_COLORS).
SECTORS: dict[str, list[str]] = {
    "Oil & Gas": ["GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SIBN", "SNGS"],
    "Metals & Mining": ["GMKN", "CHMF", "NLMK", "MAGN", "PLZL", "ALRS", "SELG", "MTLR", "RASP"],
    "Financials": ["SBER", "VTBR", "CBOM", "RENI"],
    "Power & Utilities": ["IRAO", "HYDR", "FEES", "MSNG", "UPRO"],
    "Consumer & Healthcare": ["MGNT", "BELU", "ABRD", "OZON", "MDMG"],
    "Real Estate": ["PIKK", "SMLT", "LSRG", "ETLN", "WTCM"],
    "Tech & Telecom": ["MTSS", "POSI", "WUSH"],
    "Chemicals": ["PHOR", "AKRN", "NKNC"],
    "Transport": ["FLOT", "AFLT", "FESH"],
}

TICKERS: list[str] = [t for group in SECTORS.values() for t in group]
SECTOR_OF: dict[str, str] = {t: s for s, group in SECTORS.items() for t in group}

# ── Corporate actions ────────────────────────────────────────────────────
# MOEX ISS `LEGALCLOSEPRICE` is NOT adjusted for splits. Prices strictly
# before `date` are multiplied by `factor` so that the whole series is
# expressed in post-split units. factor = old shares / new shares.
SPLITS: list[dict] = [
    {"ticker": "GMKN", "date": "2024-04-08", "factor": 1 / 100, "note": "1:100 split"},
    {"ticker": "VTBR", "date": "2024-07-15", "factor": 5000.0, "note": "5000:1 reverse split"},
    {"ticker": "BELU", "date": "2024-08-22", "factor": 1 / 8, "note": "1:8 split"},
    {"ticker": "PLZL", "date": "2025-03-27", "factor": 1 / 10, "note": "1:10 split"},
]
