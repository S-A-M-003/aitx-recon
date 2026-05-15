"""
AITX Reconciliation Engine
Matches internal vs counterparty trades, classifies breaks, scores priority.
"""

import pandas as pd
import numpy as np
import os
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QTY_TOL  = 0.0001   # 0.01%
PRC_TOL  = 0.0005   # 0.05%
PRC_P1   = 0.0010   # 0.10% — threshold above which PRC-001 escalates to P1
FAIL_BPS = 0.0002   # estimated daily cost of a settlement fail

# ---------------------------------------------------------------------------
# Taxonomy — loaded from exception_taxonomy.txt at import time
# Provides: TAXONOMY_PRIORITY, TAXONOMY_DESCRIPTION, RESOLUTION_GUIDANCE
# Dynamic overrides for STL-001 and PRC-001 are applied in _priority().
# ---------------------------------------------------------------------------
TAXONOMY_PRIORITY    = {}   # code -> base priority string ("P1"/"P2"/"P3")
TAXONOMY_DESCRIPTION = {}   # code -> short description
RESOLUTION_GUIDANCE  = {}   # code -> resolution guidance text


def load_taxonomy(path="exception_taxonomy.txt"):
    """Parse exception_taxonomy.txt and populate the three taxonomy dicts."""
    global TAXONOMY_PRIORITY, TAXONOMY_DESCRIPTION, RESOLUTION_GUIDANCE
    TAXONOMY_PRIORITY.clear()
    TAXONOMY_DESCRIPTION.clear()
    RESOLUTION_GUIDANCE.clear()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            code, priority, description, guidance = parts[0], parts[1], parts[2], parts[3]
            TAXONOMY_PRIORITY[code]    = priority
            TAXONOMY_DESCRIPTION[code] = description
            RESOLUTION_GUIDANCE[code]  = guidance

    return {
        "codes":    list(TAXONOMY_PRIORITY.keys()),
        "priority": dict(TAXONOMY_PRIORITY),
    }


def _priority(code: str, stl_days_diff=None, prc_diff_pct=None) -> str:
    """Return effective priority, applying dynamic rules for STL-001 and PRC-001."""
    base = TAXONOMY_PRIORITY.get(code, "P3")

    if code == "STL-001":
        if stl_days_diff is not None and abs(stl_days_diff) <= 1:
            return "P1"
        return "P2"

    if code == "PRC-001":
        if prc_diff_pct is not None and prc_diff_pct / 100 > PRC_P1:
            return "P1"
        return "P2"

    return base


# Load taxonomy on module import (path relative to CWD; overridable via load_taxonomy())
load_taxonomy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct_diff(a, b):
    mid = (abs(a) + abs(b)) / 2
    if mid == 0:
        return 0.0
    return abs(a - b) / mid


def _bday_diff(d1, d2):
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    d1, d2 = pd.Timestamp(d1), pd.Timestamp(d2)
    sign = 1 if d2 >= d1 else -1
    lo, hi = (d1, d2) if sign == 1 else (d2, d1)
    bdays = int(np.busday_count(lo.date(), hi.date()))
    return sign * bdays


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data(internal_path="data/internal_trades.csv",
              counterparty_path="data/counterparty_trades.csv"):
    dtype = {"trade_id": str, "isin": str, "cusip": str, "currency": str,
             "counterparty_bic": str, "account_code": str, "asset_class": str}
    parse_dates = ["settlement_date", "trade_date"]

    internal     = pd.read_csv(internal_path,     dtype=dtype, parse_dates=parse_dates)
    counterparty = pd.read_csv(counterparty_path, dtype=dtype, parse_dates=parse_dates)
    return internal, counterparty


# ---------------------------------------------------------------------------
# Post-match field validation
# ---------------------------------------------------------------------------
def _validate_pair(irow, crow):
    qty_diff  = _pct_diff(crow["quantity"], irow["quantity"])
    prc_diff  = _pct_diff(crow["price"],    irow["price"])
    stl_diff  = _bday_diff(irow["settlement_date"], crow["settlement_date"])
    isin_match = (crow["isin"] == irow["isin"])

    if not isin_match:
        return _make_result(irow, crow, "EXCEPTION", "CTY-002",
                            f"ISIN mismatch: internal={irow['isin']} cpty={crow['isin']}")
    if abs(stl_diff) >= 1:
        return _make_result(irow, crow, "EXCEPTION", "STL-001",
                            f"Settlement date differs by {stl_diff:+.0f} business day(s)")
    if prc_diff > PRC_TOL:
        return _make_result(irow, crow, "EXCEPTION", "PRC-001",
                            f"Price diff {prc_diff*100:.4f}% exceeds tolerance")
    if qty_diff > QTY_TOL:
        return _make_result(irow, crow, "EXCEPTION", "QTY-001",
                            f"Quantity diff {qty_diff*100:.4f}% exceeds tolerance")

    return _make_result(irow, crow, "MATCHED", None, None)


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------
def match_trades(internal: pd.DataFrame, counterparty: pd.DataFrame):
    matched       = []
    int_remaining = internal.copy()
    cpty_remaining = counterparty.copy()
    cpty_used = set()

    def _consume_cpty(idx):
        cpty_used.add(idx)

    def _available_cpty():
        return cpty_remaining[~cpty_remaining.index.isin(cpty_used)]

    # Pass 1: exact trade_id
    cpty_by_tid = _available_cpty().set_index("trade_id")
    int_remaining_after_p1 = []

    for _, irow in int_remaining.iterrows():
        tid = irow["trade_id"]
        if tid in cpty_by_tid.index:
            crow = cpty_by_tid.loc[tid]
            if isinstance(crow, pd.DataFrame):
                crow = crow.iloc[0]
            orig_idx = counterparty[counterparty["trade_id"] == tid].index[0]
            _consume_cpty(orig_idx)
            matched.append(_validate_pair(irow, crow))
        else:
            int_remaining_after_p1.append(irow)

    int_remaining = pd.DataFrame(int_remaining_after_p1)

    # Pass 2: ISIN + qty (tol) + settlement_date
    int_remaining_after_p2 = []

    for _, irow in int_remaining.iterrows():
        avail = _available_cpty()
        candidates = avail[
            (avail["isin"] == irow["isin"]) &
            (avail["settlement_date"] == irow["settlement_date"])
        ]
        candidates = candidates[
            candidates["quantity"].apply(
                lambda q: _pct_diff(q, irow["quantity"]) <= QTY_TOL
            )
        ]
        if not candidates.empty:
            crow = candidates.iloc[0]
            _consume_cpty(crow.name)
            matched.append(_validate_pair(irow, crow))
        else:
            int_remaining_after_p2.append(irow)

    int_remaining = pd.DataFrame(int_remaining_after_p2)

    # Pass 3: CUSIP + qty (tol) + trade_date + counterparty_bic
    int_remaining_after_p3 = []

    for _, irow in int_remaining.iterrows():
        avail = _available_cpty()
        candidates = avail[
            (avail["cusip"] == irow["cusip"]) &
            (avail["trade_date"] == irow["trade_date"]) &
            (avail["counterparty_bic"] == irow["counterparty_bic"])
        ]
        candidates = candidates[
            candidates["quantity"].apply(
                lambda q: _pct_diff(q, irow["quantity"]) <= QTY_TOL
            )
        ]
        if not candidates.empty:
            crow = candidates.iloc[0]
            _consume_cpty(crow.name)
            matched.append(_validate_pair(irow, crow))
        else:
            int_remaining_after_p3.append(irow)

    int_remaining = pd.DataFrame(int_remaining_after_p3)
    cpty_unmatched = _available_cpty()
    return matched, int_remaining, cpty_unmatched


# ---------------------------------------------------------------------------
# Break classification
# ---------------------------------------------------------------------------
def classify_breaks(int_unmatched: pd.DataFrame,
                    cpty_unmatched: pd.DataFrame,
                    internal: pd.DataFrame,
                    counterparty: pd.DataFrame):
    results  = []
    cpty_used = set()

    dup_tids = set(counterparty[counterparty.duplicated("trade_id", keep=False)]["trade_id"])
    dup_tids |= set(cpty_unmatched[cpty_unmatched["trade_id"].isin(dup_tids)]["trade_id"])

    def _available_cpty():
        return cpty_unmatched[~cpty_unmatched.index.isin(cpty_used)]

    for _, irow in int_unmatched.iterrows():
        tid = irow["trade_id"]

        if tid in dup_tids:
            dups = _available_cpty()[_available_cpty()["trade_id"] == tid]
            crow = dups.iloc[0] if not dups.empty else irow
            results.append(_make_result(irow, crow, "EXCEPTION", "DUP-001",
                                        "Duplicate trade_id in counterparty file"))
            if not dups.empty:
                cpty_used.add(dups.index[0])
            continue

        avail = _available_cpty()

        # Soft-match: ISIN + trade_date
        soft = avail[
            (avail["isin"] == irow["isin"]) &
            (avail["trade_date"] == irow["trade_date"])
        ]
        if not soft.empty:
            crow = soft.iloc[0]
            cpty_used.add(crow.name)
            qty_diff = _pct_diff(crow["quantity"], irow["quantity"])
            prc_diff = _pct_diff(crow["price"],    irow["price"])
            stl_diff = _bday_diff(irow["settlement_date"], crow["settlement_date"])

            if abs(stl_diff) >= 1:
                code   = "STL-001"
                detail = f"Settlement date differs by {stl_diff:+.0f} business day(s)"
            elif prc_diff > PRC_TOL:
                code   = "PRC-001"
                detail = f"Price diff {prc_diff*100:.4f}% exceeds tolerance"
            elif qty_diff > QTY_TOL:
                code   = "QTY-001"
                detail = f"Quantity diff {qty_diff*100:.4f}% exceeds tolerance"
            else:
                code   = "QTY-001"
                detail = "Minor field mismatch"

            results.append(_make_result(irow, crow, "EXCEPTION", code, detail))
            continue

        # Soft-match: qty + trade_date (possible CTY-002)
        soft2 = avail[
            (avail["trade_date"] == irow["trade_date"]) &
            (avail["quantity"].apply(lambda q: _pct_diff(q, irow["quantity"]) <= QTY_TOL))
        ]
        if not soft2.empty:
            crow = soft2.iloc[0]
            cpty_used.add(crow.name)
            code   = "CTY-002" if crow["isin"] != irow["isin"] else "QTY-001"
            detail = (f"ISIN mismatch: internal={irow['isin']} cpty={crow['isin']}"
                      if code == "CTY-002" else "Quantity mismatch on tertiary soft-match")
            results.append(_make_result(irow, crow, "EXCEPTION", code, detail))
            continue

        results.append(_make_result(irow, None, "EXCEPTION", "MIS-001",
                                    "Trade present in internal, absent from counterparty"))

    for _, crow in _available_cpty().iterrows():
        results.append(_make_result(None, crow, "EXCEPTION", "MIS-002",
                                    "Trade present in counterparty, absent from internal"))

    return results


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------
def _make_result(irow, crow, status, code, detail):
    trade_id = (irow["trade_id"] if irow is not None
                else crow["trade_id"] if crow is not None else None)

    qty_i = irow["quantity"] if irow is not None else np.nan
    qty_c = crow["quantity"] if crow is not None else np.nan
    prc_i = irow["price"]    if irow is not None else np.nan
    prc_c = crow["price"]    if crow is not None else np.nan
    stl_i = irow["settlement_date"] if irow is not None else pd.NaT
    stl_c = crow["settlement_date"] if crow is not None else pd.NaT

    def _safe_pct(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return round(_pct_diff(a, b) * 100, 6)

    qty_diff_pct = _safe_pct(qty_i, qty_c)
    prc_diff_pct = _safe_pct(prc_i, prc_c)
    stl_days     = _bday_diff(stl_i, stl_c)

    # P&L impact
    pnl = 0.0
    if code == "PRC-001" and not (np.isnan(qty_i) or np.isnan(prc_i) or np.isnan(prc_c)):
        pnl = abs(qty_i * (prc_i - prc_c))
    elif code == "STL-001" and not (np.isnan(qty_i) or np.isnan(prc_i)):
        pnl = abs(qty_i * prc_i * FAIL_BPS)
    elif code == "MIS-001" and not (np.isnan(qty_i) or np.isnan(prc_i)):
        pnl = abs(qty_i * prc_i * FAIL_BPS)
    elif code == "MIS-002" and not (np.isnan(qty_c) or np.isnan(prc_c)):
        pnl = abs(qty_c * prc_c * FAIL_BPS)

    # Priority
    priority = _priority(code, stl_days_diff=stl_days, prc_diff_pct=prc_diff_pct) if code else ""

    return {
        "trade_id":             trade_id,
        "status":               status,
        "exception_code":       code   or "",
        "exception_detail":     detail or "",
        "priority":             priority,
        "resolution_guidance":  RESOLUTION_GUIDANCE.get(code, "") if code else "",
        "days_open":            0,
        "quantity_diff_pct":    qty_diff_pct,
        "price_diff_pct":       prc_diff_pct,
        "settlement_days_diff": stl_days,
        "p_and_l_impact":       round(pnl, 2),
    }


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def build_summary(results_df: pd.DataFrame) -> dict:
    total     = len(results_df)
    n_matched = int((results_df["status"] == "MATCHED").sum())
    n_except  = int((results_df["status"] == "EXCEPTION").sum())
    exc_df    = results_df[results_df["status"] == "EXCEPTION"]

    return {
        "match_rate":        round(n_matched / total * 100, 4) if total else 0.0,
        "total_exceptions":  n_except,
        "p1_count":          int((exc_df["priority"] == "P1").sum()),
        "p2_count":          int((exc_df["priority"] == "P2").sum()),
        "p3_count":          int((exc_df["priority"] == "P3").sum()),
        "total_pl_at_risk":  round(float(results_df["p_and_l_impact"].sum()), 2),
        "exceptions_by_type": exc_df["exception_code"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_reconciliation(internal_path="data/internal_trades.csv",
                       counterparty_path="data/counterparty_trades.csv",
                       output_path="data/reconciliation_results.csv",
                       taxonomy_path="exception_taxonomy.txt"):

    taxonomy = load_taxonomy(taxonomy_path)
    print(f"Taxonomy loaded: {len(taxonomy['codes'])} codes from {taxonomy_path}")
    print(f"  Codes: {', '.join(taxonomy['codes'])}")

    print("\nLoading data...")
    internal, counterparty = load_data(internal_path, counterparty_path)
    print(f"  Internal trades:     {len(internal):,}")
    print(f"  Counterparty trades: {len(counterparty):,}")

    print("\nMatching trades...")
    matched, int_unmatched, cpty_unmatched = match_trades(internal, counterparty)
    print(f"  Matched (clean):     {len(matched):,}")
    print(f"  Internal unmatched:  {len(int_unmatched):,}")
    print(f"  Cpty unmatched:      {len(cpty_unmatched):,}")

    print("\nClassifying breaks...")
    breaks = classify_breaks(int_unmatched, cpty_unmatched, internal, counterparty)

    all_results = matched + breaks
    results_df  = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)

    summary = build_summary(results_df)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    total     = len(results_df)
    n_matched = int((results_df["status"] == "MATCHED").sum())
    n_except  = summary["total_exceptions"]

    print("\n" + "=" * 65)
    print("RECONCILIATION SUMMARY")
    print("=" * 65)
    print(f"  Total records processed : {total:>7,}")
    print(f"  Matched (clean)         : {n_matched:>7,}  ({summary['match_rate']:.2f}%)")
    print(f"  Exceptions              : {n_except:>7,}  ({100 - summary['match_rate']:.2f}%)")

    print(f"\n  Priority breakdown:")
    print(f"    P1 (same day)         : {summary['p1_count']:>5,}")
    print(f"    P2 (next day)         : {summary['p2_count']:>5,}")
    print(f"    P3 (by week end)      : {summary['p3_count']:>5,}")

    exc_df = results_df[results_df["status"] == "EXCEPTION"]
    print("\n  Exceptions by type (with priority and P&L):")
    print(f"  {'Code':<10} {'Count':>6}  {'Pri':<4}  {'P&L at Risk':>14}  Description")
    print("  " + "-" * 80)
    pnl_by_type = exc_df.groupby("exception_code")["p_and_l_impact"].sum()
    pri_by_type = exc_df.groupby("exception_code")["priority"].first()
    for code, cnt in sorted(summary["exceptions_by_type"].items(),
                            key=lambda x: x[1], reverse=True):
        pri  = pri_by_type.get(code, "")
        pnl  = pnl_by_type.get(code, 0.0)
        desc = TAXONOMY_DESCRIPTION.get(code, "")[:45]
        print(f"  {code:<10} {cnt:>6}  {pri:<4}  ${pnl:>13,.2f}  {desc}")

    print(f"\n  Total P&L at risk       : ${summary['total_pl_at_risk']:>12,.2f}")

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  Results saved to: {output_path}  ({size_kb:.1f} KB)")
    print("=" * 65)

    return results_df, summary


if __name__ == "__main__":
    run_reconciliation()
