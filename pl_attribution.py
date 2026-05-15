"""
AITX P&L Attribution
Calculates direct P&L, settlement fail costs, and operational costs per exception.
Produces daily_pl_summary.csv and the projected annual cost figure for client ROI.
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Cost parameters
# ---------------------------------------------------------------------------
FAIL_COST_BPS_PER_DAY = 0.0002     # 0.02% of trade value per day
OPS_RATE_GBP          = 75.0       # £/hr
OPS_HOURS = {"P1": 2.0, "P2": 1.0, "P3": 0.5}

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data(recon_path="data/reconciliation_results.csv",
              internal_path="data/internal_trades.csv"):
    recon = pd.read_csv(recon_path, dtype={"trade_id": str, "exception_code": str,
                                            "status": str, "priority": str})
    internal = pd.read_csv(internal_path, dtype={"trade_id": str},
                           parse_dates=["trade_date", "settlement_date"])
    return recon, internal


# ---------------------------------------------------------------------------
# P&L attribution
# ---------------------------------------------------------------------------
def calculate_attribution(recon: pd.DataFrame, internal: pd.DataFrame) -> pd.DataFrame:
    """
    Add cost columns to every EXCEPTION row. MATCHED rows get zeros.
    Joins with internal_trades to get quantity and price for trade value.
    """
    int_slim = internal[["trade_id", "quantity", "price", "asset_class"]].copy()
    df = recon.merge(int_slim, on="trade_id", how="left")

    df["trade_value"] = df["quantity"] * df["price"]

    exc = df["status"] == "EXCEPTION"

    # 1. Direct P&L impact -----------------------------------------------
    #    PRC-001: quantity * |price_diff|
    #    QTY-001: |qty_diff| * price  (qty_diff_pct is a %, trade_value = qty*price)
    #    Others : 0 (no direct mark-to-market impact)

    df["direct_pl_impact"] = 0.0

    prc_mask = exc & (df["exception_code"] == "PRC-001")
    df.loc[prc_mask, "direct_pl_impact"] = (
        df.loc[prc_mask, "trade_value"] *
        (df.loc[prc_mask, "price_diff_pct"].abs() / 100)
    )

    qty_mask = exc & (df["exception_code"] == "QTY-001")
    df.loc[qty_mask, "direct_pl_impact"] = (
        df.loc[qty_mask, "trade_value"] *
        (df.loc[qty_mask, "quantity_diff_pct"].abs() / 100)
    )

    # 2. Settlement fail cost ---------------------------------------------
    #    Applied to: STL-001, MIS-001, MIS-002, DUP-001
    #    Cost = trade_value * FAIL_COST_BPS_PER_DAY * |settlement_days_diff|
    #    For MIS/DUP where days_diff is 0 or NaN, use 1 day as minimum.

    FAIL_CODES = {"STL-001", "MIS-001", "MIS-002", "DUP-001"}
    fail_mask  = exc & df["exception_code"].isin(FAIL_CODES)

    days = df["settlement_days_diff"].abs().fillna(1).clip(lower=1)
    df["settlement_fail_cost"] = 0.0
    df.loc[fail_mask, "settlement_fail_cost"] = (
        df.loc[fail_mask, "trade_value"] *
        FAIL_COST_BPS_PER_DAY *
        days[fail_mask]
    )

    # 3. Operational cost -------------------------------------------------
    #    Based on priority tier; zero for MATCHED rows.

    df["ops_hours"]        = 0.0
    df["operational_cost"] = 0.0

    for priority, hours in OPS_HOURS.items():
        p_mask = exc & (df["priority"] == priority)
        df.loc[p_mask, "ops_hours"]        = hours
        df.loc[p_mask, "operational_cost"] = hours * OPS_RATE_GBP

    # 4. Total cost -------------------------------------------------------
    df["total_cost"] = (
        df["direct_pl_impact"] +
        df["settlement_fail_cost"] +
        df["operational_cost"]
    )

    # Round financial columns
    for col in ("direct_pl_impact", "settlement_fail_cost",
                "operational_cost", "total_cost", "trade_value"):
        df[col] = df[col].round(2)

    return df


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------
def build_daily_summary(attributed: pd.DataFrame, output_path="data/daily_pl_summary.csv"):
    exc = attributed[attributed["status"] == "EXCEPTION"].copy()

    # Per exception-code aggregation
    grp = exc.groupby("exception_code").agg(
        count                = ("trade_id",           "count"),
        direct_pl_impact     = ("direct_pl_impact",   "sum"),
        settlement_fail_cost = ("settlement_fail_cost","sum"),
        operational_cost     = ("operational_cost",   "sum"),
        total_cost           = ("total_cost",          "sum"),
        avg_trade_value      = ("trade_value",         "mean"),
    ).reset_index()

    grp["priority"] = exc.groupby("exception_code")["priority"].first().values

    # Grand totals row
    totals = pd.DataFrame([{
        "exception_code":       "TOTAL",
        "count":                grp["count"].sum(),
        "direct_pl_impact":     grp["direct_pl_impact"].sum(),
        "settlement_fail_cost": grp["settlement_fail_cost"].sum(),
        "operational_cost":     grp["operational_cost"].sum(),
        "total_cost":           grp["total_cost"].sum(),
        "avg_trade_value":      exc["trade_value"].mean(),
        "priority":             "",
    }])

    summary = pd.concat([grp, totals], ignore_index=True)

    # Projected annual cost
    total_daily     = grp["total_cost"].sum()
    annual_projected = total_daily * TRADING_DAYS_PER_YEAR
    summary["projected_annual_cost"] = (summary["total_cost"] * TRADING_DAYS_PER_YEAR).round(2)

    # Round
    for col in ("direct_pl_impact", "settlement_fail_cost",
                "operational_cost", "total_cost", "avg_trade_value", "projected_annual_cost"):
        summary[col] = summary[col].round(2)

    summary.to_csv(output_path, index=False)
    return summary, total_daily, annual_projected


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
def print_report(attributed: pd.DataFrame, summary: pd.DataFrame,
                 total_daily: float, annual_projected: float):

    exc = attributed[attributed["status"] == "EXCEPTION"]
    total_trades = len(attributed)
    n_matched    = (attributed["status"] == "MATCHED").sum()
    n_exceptions = len(exc)

    print("\n" + "=" * 70)
    print("  AITX P&L ATTRIBUTION REPORT")
    print("=" * 70)

    print(f"\n  Portfolio snapshot")
    print(f"  {'Total trades processed':<35} {total_trades:>10,}")
    print(f"  {'Matched (clean)':<35} {n_matched:>10,}  ({n_matched/total_trades*100:.1f}%)")
    print(f"  {'Exceptions':<35} {n_exceptions:>10,}  ({n_exceptions/total_trades*100:.1f}%)")

    # Cost breakdown
    direct  = exc["direct_pl_impact"].sum()
    fail    = exc["settlement_fail_cost"].sum()
    ops     = exc["operational_cost"].sum()
    total   = exc["total_cost"].sum()
    ops_hrs = exc["ops_hours"].sum()

    print(f"\n  Cost breakdown (today's exceptions)")
    print(f"  {'Direct P&L impact (price/qty breaks)':<45} GBP {direct:>12,.2f}")
    print(f"  {'Settlement fail costs (@ 0.02%/day)':<45} GBP {fail:>12,.2f}")
    print(f"  {'Operational cost (' + f'{ops_hrs:.0f}' + ' hrs @ GBP75/hr)':<45} GBP {ops:>12,.2f}")
    print(f"  {'-'*55}")
    print(f"  {'Total cost (single day)':<45} GBP {total:>12,.2f}")

    # By exception type
    print(f"\n  {'Code':<10} {'Pri':<4} {'Count':>6}  {'Direct P&L':>12}  "
          f"{'Fail Cost':>12}  {'Ops Cost':>9}  {'Total':>12}")
    print(f"  {'-'*73}")
    disp = summary[summary["exception_code"] != "TOTAL"].sort_values("total_cost", ascending=False)
    for _, row in disp.iterrows():
        print(f"  {row['exception_code']:<10} {row['priority']:<4} {int(row['count']):>6}  "
              f"GBP {row['direct_pl_impact']:>9,.0f}  "
              f"GBP {row['settlement_fail_cost']:>9,.0f}  "
              f"GBP {row['operational_cost']:>6,.0f}  "
              f"GBP {row['total_cost']:>9,.0f}")

    # Priority breakdown
    print(f"\n  Cost by priority tier")
    for pri in ("P1", "P2", "P3"):
        subset = exc[exc["priority"] == pri]
        if subset.empty:
            continue
        print(f"  {pri}  {len(subset):>4} exceptions  "
              f"GBP {subset['total_cost'].sum():>12,.2f}  "
              f"({subset['ops_hours'].sum():.0f} ops-hours)")

    # Asset class breakdown
    print(f"\n  Cost by asset class")
    for ac, grp in exc.groupby("asset_class"):
        print(f"  {ac:<15}  {len(grp):>4} exceptions  GBP {grp['total_cost'].sum():>12,.2f}")

    # -----------------------------------------------------------------------
    # ROI banner — annual projection
    # -----------------------------------------------------------------------
    print()
    print("  " + "=" * 66)
    print("  *** PROJECTED ANNUAL COST OF UNRESOLVED EXCEPTIONS ***")
    print("  " + "=" * 66)
    print()
    print(f"       Daily exception cost   :  GBP {total_daily:>14,.2f}")
    print(f"       Trading days per year  :         {TRADING_DAYS_PER_YEAR}")
    print()
    print(f"  +---------------------------------------------------------+")
    print(f"  |   PROJECTED ANNUAL COST :  GBP {annual_projected:>14,.2f}          |")
    print(f"  +---------------------------------------------------------+")
    print()
    print(f"  This represents the financial exposure from today's {n_exceptions}")
    print(f"  exceptions if left unresolved at current exception rates.")
    print(f"  AITX auto-resolution targets 80%+ of P2/P3 exceptions,")

    atr_exc = exc[exc["priority"].isin(["P2", "P3"])]
    atr_cost = atr_exc["total_cost"].sum() * 0.80
    atr_annual = atr_cost * TRADING_DAYS_PER_YEAR
    print(f"  saving an estimated GBP {atr_annual:,.0f}/year in this portfolio.")
    print("  " + "=" * 66)

    print(f"\n  Files saved:")
    print(f"    data/daily_pl_summary.csv")
    print(f"    data/reconciliation_results.csv  (updated with cost columns)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    recon, internal = load_data()

    print("Calculating P&L attribution...")
    attributed = calculate_attribution(recon, internal)

    # Persist cost columns back to reconciliation_results.csv
    attributed.drop(columns=["quantity", "price", "asset_class", "trade_value"],
                    errors="ignore").to_csv("data/reconciliation_results.csv", index=False)

    # Keep trade_value in attributed for summary grouping
    print("Building daily summary...")
    summary, total_daily, annual_projected = build_daily_summary(attributed)

    print_report(attributed, summary, total_daily, annual_projected)
