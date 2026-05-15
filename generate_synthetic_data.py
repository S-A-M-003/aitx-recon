import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta
import random
import string
import os

fake = Faker()
rng = np.random.default_rng(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
ASSET_CLASSES = ["FX_SPOT", "FX_FORWARD", "EQUITY", "FIXED_INCOME"]
ASSET_WEIGHTS = [0.20, 0.20, 0.35, 0.25]

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD"]

def random_isin():
    country = random.choice(["US", "GB", "DE", "FR", "JP", "AU", "CA"])
    return country + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

def random_cusip():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=9))

def random_bic():
    bank = "".join(random.choices(string.ascii_uppercase, k=4))
    country = random.choice(["US", "GB", "DE", "FR", "JP", "AU", "CA"])
    location = "".join(random.choices(string.ascii_uppercase + string.digits, k=2))
    branch = "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
    return f"{bank}{country}{location}{branch}"

def random_account_code():
    prefix = random.choice(["ACC", "TRD", "CLR", "CUS"])
    return f"{prefix}-{random.randint(10000, 99999)}"

def next_business_day(d, delta_days):
    result = d
    step = 1 if delta_days > 0 else -1
    remaining = abs(delta_days)
    while remaining > 0:
        result += timedelta(days=step)
        if result.weekday() < 5:
            remaining -= 1
    return result

def random_trade_date():
    start = date(2024, 1, 1)
    end = date(2025, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def settlement_date_for(trade_date, asset_class):
    offset = {"FX_SPOT": 2, "FX_FORWARD": random.randint(30, 180),
              "EQUITY": 2, "FIXED_INCOME": 1}[asset_class]
    return next_business_day(trade_date, offset)

def random_price(asset_class):
    if asset_class in ("FX_SPOT", "FX_FORWARD"):
        return round(random.uniform(0.5, 2.5), 6)
    elif asset_class == "EQUITY":
        return round(random.uniform(5.0, 500.0), 4)
    else:
        return round(random.uniform(85.0, 115.0), 4)

def random_quantity(asset_class):
    if asset_class in ("FX_SPOT", "FX_FORWARD"):
        return round(random.uniform(100_000, 10_000_000), 2)
    elif asset_class == "EQUITY":
        return float(random.randint(100, 50_000))
    else:
        return float(random.randint(1_000, 1_000_000) // 1000 * 1000)


# ---------------------------------------------------------------------------
# Generate base records
# ---------------------------------------------------------------------------
N = 10_000
BREAK_FRACTION = 0.05
N_BREAKS = int(N * BREAK_FRACTION)   # 500 breaks
N_CLEAN = N - N_BREAKS               # 9500 clean

asset_classes = random.choices(ASSET_CLASSES, weights=ASSET_WEIGHTS, k=N)

records = []
for i in range(N):
    ac = asset_classes[i]
    td = random_trade_date()
    sd = settlement_date_for(td, ac)
    records.append({
        "trade_id":        f"TRD-{i+1:06d}",
        "isin":            random_isin(),
        "cusip":           random_cusip(),
        "quantity":        random_quantity(ac),
        "price":           random_price(ac),
        "currency":        random.choice(CURRENCIES),
        "settlement_date": sd,
        "counterparty_bic":random_bic(),
        "account_code":    random_account_code(),
        "trade_date":      td,
        "asset_class":     ac,
    })

base_df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Assign break types
# ---------------------------------------------------------------------------
BREAK_TYPES = ["QTY-001", "STL-001", "PRC-001", "CTY-002", "MIS-001", "DUP-001"]
break_indices = random.sample(range(N), N_BREAKS)
break_assignments = {idx: BREAK_TYPES[i % len(BREAK_TYPES)] for i, idx in enumerate(break_indices)}

# ---------------------------------------------------------------------------
# Build internal and counterparty DataFrames
# ---------------------------------------------------------------------------
internal_rows = []
counterparty_rows = []
break_label_rows = []

for i, row in base_df.iterrows():
    int_row = row.copy()
    cpty_row = row.copy()
    label = "CLEAN"

    if i in break_assignments:
        btype = break_assignments[i]
        label = btype

        if btype == "QTY-001":
            # quantity mismatch > 0.01%
            factor = 1.0 + random.choice([-1, 1]) * random.uniform(0.0002, 0.005)
            cpty_row["quantity"] = round(row["quantity"] * factor, 2)

        elif btype == "STL-001":
            # settlement date off by 1-2 business days
            delta = random.choice([-2, -1, 1, 2])
            cpty_row["settlement_date"] = next_business_day(row["settlement_date"], delta)

        elif btype == "PRC-001":
            # price difference > 0.05%
            factor = 1.0 + random.choice([-1, 1]) * random.uniform(0.0006, 0.003)
            cpty_row["price"] = round(row["price"] * factor, 6)

        elif btype == "CTY-002":
            # different ISIN
            cpty_row["isin"] = random_isin()

        elif btype == "MIS-001":
            # trade present internally but missing from counterparty
            internal_rows.append(int_row)
            break_label_rows.append({"trade_id": row["trade_id"], "break_type": label})
            continue  # skip adding to counterparty

        elif btype == "DUP-001":
            # duplicate entry in counterparty
            counterparty_rows.append(cpty_row)
            counterparty_rows.append(cpty_row.copy())  # second copy = duplicate
            internal_rows.append(int_row)
            break_label_rows.append({"trade_id": row["trade_id"], "break_type": label})
            continue

    internal_rows.append(int_row)
    counterparty_rows.append(cpty_row)
    break_label_rows.append({"trade_id": row["trade_id"], "break_type": label})

internal_df = pd.DataFrame(internal_rows)
counterparty_df = pd.DataFrame(counterparty_rows)
labels_df = pd.DataFrame(break_label_rows)

# ---------------------------------------------------------------------------
# Save files
# ---------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)

internal_df.to_csv("data/internal_trades.csv", index=False)
counterparty_df.to_csv("data/counterparty_trades.csv", index=False)
labels_df.to_csv("data/break_labels.csv", index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("SYNTHETIC DATA GENERATION SUMMARY")
print("=" * 60)
print(f"\nInternal trades:      {len(internal_df):>7,}")
print(f"Counterparty trades:  {len(counterparty_df):>7,}")
print(f"Break label records:  {len(labels_df):>7,}")

print("\nBreak type distribution:")
bc = labels_df["break_type"].value_counts()
for btype, count in bc.items():
    pct = count / len(labels_df) * 100
    print(f"  {btype:<12} {count:>5,}  ({pct:.1f}%)")

print("\nAsset class distribution (internal):")
ac = internal_df["asset_class"].value_counts()
for aclass, count in ac.items():
    print(f"  {aclass:<15} {count:>5,}")

print("\nCurrency distribution (internal):")
cc = internal_df["currency"].value_counts()
for curr, count in cc.items():
    print(f"  {curr:<8} {count:>5,}")

print("\nDate range:")
print(f"  Trade date:      {internal_df['trade_date'].min()} to {internal_df['trade_date'].max()}")
print(f"  Settlement date: {internal_df['settlement_date'].min()} to {internal_df['settlement_date'].max()}")

print("\nFiles saved:")
for f in ["data/internal_trades.csv", "data/counterparty_trades.csv", "data/break_labels.csv"]:
    size_kb = os.path.getsize(f) / 1024
    print(f"  {f:<35} {size_kb:>8.1f} KB")

print("=" * 60)
