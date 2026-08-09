"""
Synthetic pick-up point network data for the commission grain analysis.

Entity hierarchy:

    order    ->  one or more POSTINGS  ->  one or more ITEMS
                 (a posting is a physical box and cannot be split)

The platform pays each pick-up point partner a commission on every ITEM
handed over, as a percentage of item value, capped PER ITEM. The proposed
change moves commission to POSTING grain: same rate, cap applied once per
box. Because a box can hold several items, collapsing the cap is where any
saving comes from -- and the size of that saving depends on how many items
travel in a box, which differs by fulfilment channel.

Planted data-quality issues, mirroring real transactional warehouses:
  - return rows that must be netted, not double-counted
  - superseded ("corrected") report versions that must be excluded
  - orders arriving as several postings on different days
  - a small number of orphaned returns with no matching handover

Output: fct_posting_items.parquet, dim_pickup_points.parquet

Run:  python generate_data.py
"""

import numpy as np
import pandas as pd

SEED = 20260215
N_ORDERS = 60_000
PERIOD_START = pd.Timestamp("2026-02-01")
PERIOD_DAYS = 28

# Current schedule: percentage of item value, capped PER ITEM.
COMMISSION_RATE = 0.055
CAP_PER_ITEM = 2.50

rng = np.random.default_rng(SEED)


# -------------------------------------------------------- pick-up locations

def make_pickup_points(n=850):
    """Partner-operated locations, grouped by how large the operator's network is."""
    network_tier = rng.choice(
        ["single_site", "small_network", "large_network"],
        size=n, p=[0.55, 0.32, 0.13],
    )
    return pd.DataFrame({
        "pickup_point_id": [f"P{str(i).zfill(5)}" for i in range(1, n + 1)],
        "network_tier": network_tier,
        "region_type": rng.choice(
            ["metro", "regional_city", "small_town"],
            size=n, p=[0.34, 0.41, 0.25],
        ),
    })


# ---------------------------------------------------------------- channels

# Fulfilment channel determines how goods are consolidated into boxes, and
# therefore how many items travel per posting. These profiles describe how
# fulfilment physically works -- they are NOT set to produce a chosen answer.
# Each channel's share of value versus share of postings is emergent, and is
# one of the things the analysis exists to discover.
CHANNEL_PROFILE = {
    # platform warehouse: consolidated picking, many low-value items per box
    "warehouse":        {"order_share": 0.72, "postings_lam": 1.25,
                         "items_lam": 3.6, "value_mu": 0.60, "value_sd": 1.30},
    # seller ships direct: usually one box, fewer and pricier items
    "merchant_shipped": {"order_share": 0.13, "postings_lam": 1.08,
                         "items_lam": 1.5, "value_mu": 2.18, "value_sd": 1.34},
    # store-fulfilled: mid-sized boxes
    "retail":           {"order_share": 0.06, "postings_lam": 1.10,
                         "items_lam": 2.2, "value_mu": 1.27, "value_sd": 1.31},
    # imported goods: small boxes, higher unit value
    "cross_border":     {"order_share": 0.09, "postings_lam": 1.15,
                         "items_lam": 1.8, "value_mu": 1.81, "value_sd": 1.33},
}

CHANNELS = list(CHANNEL_PROFILE)
CHANNEL_P = [CHANNEL_PROFILE[c]["order_share"] for c in CHANNELS]


def make_orders(points):
    channel = rng.choice(CHANNELS, size=N_ORDERS, p=CHANNEL_P)

    lam = np.array([CHANNEL_PROFILE[c]["postings_lam"] for c in channel])
    n_postings = np.clip(rng.poisson(lam - 1) + 1, 1, 5)

    return pd.DataFrame({
        "order_id": [f"O{str(i).zfill(7)}" for i in range(1, N_ORDERS + 1)],
        "pickup_point_id": rng.choice(points["pickup_point_id"], size=N_ORDERS),
        "fulfilment_channel": channel,
        "n_postings": n_postings,
        "order_date": PERIOD_START + pd.to_timedelta(
            rng.integers(0, PERIOD_DAYS, size=N_ORDERS), unit="D"),
    })


# ------------------------------------------------------------- postings

def make_postings(orders):
    """Explode orders into physical boxes. Boxes of one order can arrive on
    different days, which is why the analysis cannot simply work at order grain."""
    rep = orders.loc[orders.index.repeat(orders["n_postings"])].reset_index(drop=True)
    seq = rep.groupby("order_id").cumcount() + 1

    day_gap = np.where(seq > 1, rng.integers(1, 4, size=len(rep)), 0)

    lam = rep["fulfilment_channel"].map(
        lambda c: CHANNEL_PROFILE[c]["items_lam"]).to_numpy()
    n_items = np.clip(rng.poisson(lam - 1) + 1, 1, 30)

    return pd.DataFrame({
        "posting_id": rep["order_id"] + "-B" + seq.astype(str),
        "order_id": rep["order_id"],
        "pickup_point_id": rep["pickup_point_id"],
        "fulfilment_channel": rep["fulfilment_channel"],
        "handover_date": rep["order_date"] + pd.to_timedelta(day_gap, unit="D"),
        "n_items": n_items,
    })


# ---------------------------------------------------------------- items

def make_items(postings):
    rep = postings.loc[postings.index.repeat(postings["n_items"])].reset_index(drop=True)
    n = len(rep)

    mu = rep["fulfilment_channel"].map(
        lambda c: CHANNEL_PROFILE[c]["value_mu"]).to_numpy()
    sd = rep["fulfilment_channel"].map(
        lambda c: CHANNEL_PROFILE[c]["value_sd"]).to_numpy()

    item_value = np.round(np.clip(np.exp(rng.normal(mu, sd)), 0.99, 4000.0), 2)

    return pd.DataFrame({
        "posting_item_id": [f"I{str(i).zfill(8)}" for i in range(1, n + 1)],
        "posting_id": rep["posting_id"],
        "order_id": rep["order_id"],
        "pickup_point_id": rep["pickup_point_id"],
        "fulfilment_channel": rep["fulfilment_channel"],
        "handover_date": rep["handover_date"],
        "event_type": "handover",
        "record_status": "final",
        "item_value": item_value,
    })


# ---------------------------------------------------------- planted defects

def add_returns(items, rate=0.035):
    """Returned items. Mirror rows with a negative-signed event type."""
    picked = items.sample(frac=rate, random_state=SEED)
    rev = picked.copy()
    rev["posting_item_id"] = "R" + rev["posting_item_id"].str[1:]
    rev["event_type"] = "handover_return"
    rev["handover_date"] = rev["handover_date"] + pd.to_timedelta(
        rng.integers(0, 6, size=len(rev)), unit="D")
    return pd.concat([items, rev], ignore_index=True)


def add_orphan_returns(items, n=180):
    """Returns whose original handover falls outside the reporting period."""
    orphan = items.sample(n=n, random_state=SEED + 1).copy()
    orphan["posting_item_id"] = "X" + orphan["posting_item_id"].str[1:]
    orphan["posting_id"] = "O9" + orphan["posting_id"].str[2:]
    orphan["event_type"] = "handover_return"
    return pd.concat([items, orphan], ignore_index=True)


def add_corrected_versions(items, rate=0.02):
    """Restated agent-report rows. The superseded version must be excluded."""
    picked = items[items["event_type"] == "handover"].sample(
        frac=rate, random_state=SEED + 2)
    old = picked.copy()
    old["record_status"] = "corrected"
    old["item_value"] = np.round(
        old["item_value"] * rng.uniform(0.8, 1.2, len(old)), 2)
    old["posting_item_id"] = "C" + old["posting_item_id"].str[1:]
    return pd.concat([items, old], ignore_index=True)


def add_current_commission(items):
    """Current schedule: percentage of item value, capped PER ITEM."""
    sign = np.where(items["event_type"].str.endswith("return"), -1, 1)
    gross = np.minimum(items["item_value"].abs() * COMMISSION_RATE, CAP_PER_ITEM)
    items["item_commission"] = np.round(gross * sign, 4)
    items["item_value"] = np.round(items["item_value"].abs() * sign, 2)
    return items


# --------------------------------------------------------------------- main

def main():
    points = make_pickup_points()
    orders = make_orders(points)
    postings = make_postings(orders)

    items = make_items(postings)
    items = add_returns(items)
    items = add_orphan_returns(items)
    items = add_corrected_versions(items)
    items = add_current_commission(items)

    items = items.sample(frac=1.0, random_state=SEED + 3).reset_index(drop=True)
    items = items[[
        "posting_item_id", "posting_id", "order_id", "pickup_point_id",
        "fulfilment_channel", "handover_date", "event_type", "record_status",
        "item_value", "item_commission",
    ]]

    items.to_parquet("fct_posting_items.parquet", index=False)
    points.to_parquet("dim_pickup_points.parquet", index=False)
    items.head(2000).to_csv("sample_preview.csv", index=False)

    print(f"rows                : {len(items):,}")
    print(f"distinct postings   : {items['posting_id'].nunique():,}")
    print(f"distinct orders     : {items['order_id'].nunique():,}")
    print(f"pick-up points      : {items['pickup_point_id'].nunique():,}")
    print(f"return rows         : {(items.event_type == 'handover_return').sum():,}")
    print(f"corrected rows      : {(items.record_status == 'corrected').sum():,}")
    print(f"gross value         : {items['item_value'].sum():,.0f}")
    print(f"commission paid     : {items['item_commission'].sum():,.0f}")
    print()

    live = items[(items.record_status != "corrected")
                 & (items.event_type == "handover")]
    mix = live.groupby("fulfilment_channel").agg(
        postings=("posting_id", "nunique"),
        value=("item_value", "sum"),
        items=("posting_item_id", "count"),
    )
    mix["pct_postings"] = (100 * mix.postings / mix.postings.sum()).round(1)
    mix["pct_value"] = (100 * mix.value / mix.value.sum()).round(1)
    mix["items_per_posting"] = (mix["items"] / mix.postings).round(2)
    over = live[live.item_value > 50]
    print(f"items over 50     : {100*len(over)/len(live):.2f}% of items, "
          f"{100*over.item_value.sum()/live.item_value.sum():.1f}% of value")
    print()
    print("channel mix (emergent, not set directly):")
    print(mix[["pct_postings", "pct_value", "items_per_posting"]].to_string())


if __name__ == "__main__":
    main()
