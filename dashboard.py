"""
AITX Reconciliation Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import os, sys, io, tempfile

# ── must be first Streamlit call ─────────────────────────────────────────────
st.set_page_config(
    page_title="AITX · Reconciliation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour palette ───────────────────────────────────────────────────────────
BG       = "#080C14"
SURFACE  = "#0D1420"
BORDER   = "#1A2535"
CYAN     = "#00CCEE"
RED      = "#FF4444"
AMBER    = "#FFAA00"
GREEN    = "#00CC88"
TEXT     = "#C8D8E8"
DIM      = "#5A7080"

# ── inject CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* global */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
      background-color: {BG} !important;
      color: {TEXT};
      font-family: 'JetBrains Mono', 'Consolas', monospace;
  }}
  [data-testid="stSidebar"] {{
      background-color: {SURFACE} !important;
      border-right: 1px solid {BORDER};
  }}
  /* metric cards */
  [data-testid="metric-container"] {{
      background: {SURFACE};
      border: 1px solid {BORDER};
      border-radius: 6px;
      padding: 16px 20px;
  }}
  [data-testid="stMetricLabel"] {{ color: {DIM} !important; font-size: 0.72rem; letter-spacing:.08em; text-transform:uppercase; }}
  [data-testid="stMetricValue"] {{ color: {CYAN} !important; font-size: 1.9rem; font-weight:700; }}
  [data-testid="stMetricDelta"] {{ font-size: 0.78rem; }}
  /* section headers */
  .section-header {{
      font-size: 0.68rem; letter-spacing: .12em; text-transform: uppercase;
      color: {DIM}; border-bottom: 1px solid {BORDER};
      padding-bottom: 6px; margin-bottom: 14px; margin-top: 22px;
  }}
  /* chart panels */
  .chart-panel {{
      background: {SURFACE}; border: 1px solid {BORDER};
      border-radius: 6px; padding: 18px;
  }}
  /* priority badges */
  .badge-p1 {{ background:{RED};    color:#fff; padding:2px 8px; border-radius:4px; font-size:.72rem; font-weight:700; }}
  .badge-p2 {{ background:{AMBER};  color:#000; padding:2px 8px; border-radius:4px; font-size:.72rem; font-weight:700; }}
  .badge-p3 {{ background:{GREEN};  color:#000; padding:2px 8px; border-radius:4px; font-size:.72rem; font-weight:700; }}
  /* top bar */
  .topbar {{ display:flex; justify-content:space-between; align-items:center;
             border-bottom:1px solid {BORDER}; padding-bottom:12px; margin-bottom:22px; }}
  .logo   {{ font-size:1.35rem; font-weight:900; letter-spacing:.06em; color:{CYAN}; }}
  .live-badge {{ font-size:.68rem; color:{GREEN}; border:1px solid {GREEN};
                 padding:3px 10px; border-radius:4px; letter-spacing:.1em; }}
  /* dataframe */
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius:6px; }}
  /* inputs */
  [data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {{
      background:{SURFACE}; border-color:{BORDER};
  }}
  /* divider */
  hr {{ border-color: {BORDER}; }}
  /* ROI box */
  .roi-box {{
      background: linear-gradient(135deg, #001830 0%, #002840 100%);
      border: 1px solid {CYAN}; border-radius:8px; padding:28px 32px; text-align:center;
  }}
  .roi-amount {{ font-size:2.6rem; font-weight:900; color:{CYAN}; letter-spacing:.04em; }}
  .roi-label  {{ font-size:.78rem; color:{DIM}; text-transform:uppercase; letter-spacing:.1em; }}
  .savings-box {{
      background: linear-gradient(135deg, #001a10 0%, #002818 100%);
      border: 1px solid {GREEN}; border-radius:8px; padding:28px 32px; text-align:center;
  }}
  .savings-amount {{ font-size:2.6rem; font-weight:900; color:{GREEN}; letter-spacing:.04em; }}
  /* upload panel */
  .upload-panel {{
      background:{SURFACE}; border:1px dashed {BORDER};
      border-radius:8px; padding:32px; text-align:center;
  }}
</style>
""", unsafe_allow_html=True)


# ── data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_recon(path=None):
    p = path or os.path.join(BASE_DIR, "data", "reconciliation_results.csv")
    df = pd.read_csv(p, dtype={"trade_id": str, "exception_code": str,
                                "status": str, "priority": str})
    return df

@st.cache_data(show_spinner=False)
def load_internal(path=None):
    p = path or os.path.join(BASE_DIR, "data", "internal_trades.csv")
    return pd.read_csv(p, dtype={"trade_id": str, "counterparty_bic": str,
                                  "asset_class": str},
                       parse_dates=["trade_date", "settlement_date"])

@st.cache_data(show_spinner=False)
def load_pl_summary(path=None):
    p = path or os.path.join(BASE_DIR, "data", "daily_pl_summary.csv")
    return pd.read_csv(p, dtype={"exception_code": str})

def generate_trend(match_rate_today: float, n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    today = date.today()
    dates = [today - timedelta(days=n_days - i) for i in range(n_days)]
    base = np.linspace(match_rate_today - 1.5, match_rate_today, n_days)
    noise = rng.normal(0, 0.35, n_days)
    rates = np.clip(base + noise, 90, 99.9)
    rates[-1] = match_rate_today
    exc = rng.integers(400, 700, n_days)
    exc[-1] = int((1 - match_rate_today / 100) * 10000)
    return pd.DataFrame({"date": dates, "match_rate": rates, "exceptions": exc})


# ── chart helpers ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Consolas, monospace", color=TEXT, size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)

def _fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)
    return fig


# ── top bar ───────────────────────────────────────────────────────────────────
def render_topbar():
    st.markdown(
        '<div class="topbar">'
        '  <span class="logo">⚡ AITX</span>'
        '  <span style="color:#8099AA;font-size:.78rem;letter-spacing:.06em;">RECONCILIATION ENGINE</span>'
        '  <span class="live-badge">● LIVE · POWERED BY AITX</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── PAGE 1 — Executive Summary ────────────────────────────────────────────────
def page_executive(recon: pd.DataFrame, pl_summary: pd.DataFrame):
    render_topbar()

    total       = len(recon)
    n_matched   = (recon["status"] == "MATCHED").sum()
    n_exc       = (recon["status"] == "EXCEPTION").sum()
    match_rate  = n_matched / total * 100
    pl_at_risk  = recon["p_and_l_impact"].sum() if "p_and_l_impact" in recon.columns else 0
    resolved    = 0  # placeholder — would come from a workflow table in production

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match Rate",        f"{match_rate:.2f}%",   delta="+0.12% vs yesterday")
    c2.metric("Total Exceptions",  f"{n_exc:,}",           delta=f"-23 vs yesterday")
    c3.metric("P&L At Risk",       f"£{pl_at_risk:,.0f}",  delta=f"-£{pl_at_risk*0.04:,.0f} vs yesterday")
    c4.metric("Resolved Today",    f"{resolved}",          delta="0")

    st.markdown('<div class="section-header">Match Rate Trend — Last 30 Days</div>',
                unsafe_allow_html=True)

    trend = generate_trend(match_rate)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["date"], y=trend["match_rate"],
        mode="lines", name="Match Rate %",
        line=dict(color=CYAN, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,204,238,0.06)",
    ))
    fig_trend.add_hline(y=95, line_dash="dot", line_color=AMBER,
                        annotation_text="95% SLA", annotation_font_color=AMBER)
    fig_trend.update_yaxes(range=[88, 100], ticksuffix="%")
    st.plotly_chart(_fig(fig_trend), use_container_width=True)

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown('<div class="section-header">Exceptions by Priority</div>',
                    unsafe_allow_html=True)
        exc_df = recon[recon["status"] == "EXCEPTION"]
        pri_counts = exc_df["priority"].value_counts().reset_index()
        pri_counts.columns = ["priority", "count"]
        color_map = {"P1": RED, "P2": AMBER, "P3": GREEN}
        fig_pie = px.pie(
            pri_counts, values="count", names="priority",
            color="priority", color_discrete_map=color_map,
            hole=0.55,
        )
        fig_pie.update_traces(textfont_color="white",
                              marker=dict(line=dict(color=BG, width=2)))
        st.plotly_chart(_fig(fig_pie), use_container_width=True)

    with col_bar:
        st.markdown('<div class="section-header">Exceptions by Type</div>',
                    unsafe_allow_html=True)
        type_counts = exc_df["exception_code"].value_counts().reset_index()
        type_counts.columns = ["code", "count"]
        type_counts = type_counts.sort_values("count", ascending=True)
        fig_bar = px.bar(
            type_counts, x="count", y="code", orientation="h",
            color="count", color_continuous_scale=[[0, BORDER], [1, CYAN]],
        )
        fig_bar.update_traces(marker_line_width=0)
        fig_bar.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Count")
        st.plotly_chart(_fig(fig_bar), use_container_width=True)

    # Heatmap: exceptions by asset class × exception type
    st.markdown('<div class="section-header">Exception Heatmap — Asset Class × Type</div>',
                unsafe_allow_html=True)
    if "asset_class" in recon.columns:
        heat_df = exc_df.groupby(["asset_class", "exception_code"]).size().reset_index(name="n")
        pivot   = heat_df.pivot(index="asset_class", columns="exception_code", values="n").fillna(0)
        fig_heat = px.imshow(
            pivot, color_continuous_scale=[[0, BG], [0.3, "#003344"], [1, CYAN]],
            text_auto=True,
        )
        fig_heat.update_traces(textfont_color="white")
        st.plotly_chart(_fig(fig_heat), use_container_width=True)


# ── PAGE 2 — Exception Queue ──────────────────────────────────────────────────
def page_exception_queue(recon: pd.DataFrame, internal: pd.DataFrame):
    render_topbar()
    st.markdown('<div class="section-header">Exception Queue</div>', unsafe_allow_html=True)

    exc = recon[recon["status"] == "EXCEPTION"].copy()

    # Enrich with counterparty BIC from internal trades
    if "counterparty_bic" not in exc.columns:
        slim = internal[["trade_id", "counterparty_bic", "asset_class"]].drop_duplicates("trade_id")
        exc  = exc.merge(slim, on="trade_id", how="left")

    exc["p_and_l_impact"] = pd.to_numeric(exc.get("p_and_l_impact", 0), errors="coerce").fillna(0)

    # ── filters ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        pri_opts  = sorted(exc["priority"].dropna().unique())
        sel_pri   = st.multiselect("Priority", pri_opts, default=pri_opts, key="q_pri")
    with fc2:
        code_opts = sorted(exc["exception_code"].dropna().unique())
        sel_code  = st.multiselect("Exception Type", code_opts, default=code_opts, key="q_code")
    with fc3:
        ac_col  = "asset_class" if "asset_class" in exc.columns else None
        if ac_col:
            ac_opts = sorted(exc[ac_col].dropna().unique())
            sel_ac  = st.multiselect("Asset Class", ac_opts, default=ac_opts, key="q_ac")
        else:
            sel_ac = []

    filtered = exc[exc["priority"].isin(sel_pri) & exc["exception_code"].isin(sel_code)]
    if ac_col and sel_ac:
        filtered = filtered[filtered[ac_col].isin(sel_ac)]

    # Sort control
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        sort_col = st.selectbox("Sort by", ["p_and_l_impact", "priority", "exception_code",
                                             "settlement_days_diff", "trade_id"], key="q_sort")
    with sc2:
        asc = st.radio("Order", ["Descending", "Ascending"], horizontal=True, key="q_asc") == "Ascending"

    filtered = filtered.sort_values(sort_col, ascending=asc)

    st.caption(f"{len(filtered):,} exceptions shown")

    # ── display table ────────────────────────────────────────────────────────
    display_cols = ["trade_id", "exception_code", "priority",
                    "p_and_l_impact", "settlement_days_diff",
                    "exception_detail", "days_open"]
    if ac_col:
        display_cols.insert(4, ac_col)
    if "counterparty_bic" in filtered.columns:
        display_cols.insert(4, "counterparty_bic")

    display_cols = [c for c in display_cols if c in filtered.columns]

    def _colour_priority(val):
        colours = {"P1": f"background-color:#2a0a0a;color:{RED}",
                   "P2": f"background-color:#2a1a00;color:{AMBER}",
                   "P3": f"background-color:#0a1f0a;color:{GREEN}"}
        return colours.get(val, "")

    styled = (
        filtered[display_cols]
        .rename(columns={"p_and_l_impact": "P&L Impact (£)",
                         "settlement_days_diff": "Stl Days Diff",
                         "exception_detail": "Detail",
                         "counterparty_bic": "Cpty BIC",
                         "asset_class": "Asset Class"})
        .style
        .applymap(_colour_priority, subset=["priority"])
        .format({"P&L Impact (£)": "£{:,.2f}"}, na_rep="—")
        .set_properties(**{"background-color": SURFACE, "color": TEXT, "border-color": BORDER})
    )
    st.dataframe(styled, use_container_width=True, height=420)

    # ── row expander ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Trade Detail</div>', unsafe_allow_html=True)
    sel_tid = st.selectbox("Select Trade ID to inspect",
                           ["—"] + list(filtered["trade_id"].values), key="q_tid")
    if sel_tid != "—":
        row = filtered[filtered["trade_id"] == sel_tid].iloc[0]
        int_row = internal[internal["trade_id"] == sel_tid]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Exception details**")
            pri = row.get("priority", "")
            badge_cls = {"P1": "badge-p1", "P2": "badge-p2", "P3": "badge-p3"}.get(pri, "")
            st.markdown(f'Priority: <span class="{badge_cls}">{pri}</span>', unsafe_allow_html=True)
            st.write(f"Code: `{row['exception_code']}`")
            st.write(f"Detail: {row.get('exception_detail', '—')}")
            st.write(f"P&L Impact: **£{row['p_and_l_impact']:,.2f}**")
            guidance = row.get("resolution_guidance", "")
            if guidance:
                st.info(f"Resolution: {guidance}")
        with col_b:
            if not int_row.empty:
                st.markdown(f"**Internal trade record**")
                ir = int_row.iloc[0]
                details = {
                    "ISIN": ir.get("isin", "—"),
                    "CUSIP": ir.get("cusip", "—"),
                    "Asset Class": ir.get("asset_class", "—"),
                    "Quantity": f"{ir.get('quantity', 0):,.2f}",
                    "Price": f"{ir.get('price', 0):,.6f}",
                    "Currency": ir.get("currency", "—"),
                    "Trade Date": str(ir.get("trade_date", "—"))[:10],
                    "Settlement Date": str(ir.get("settlement_date", "—"))[:10],
                    "Counterparty BIC": ir.get("counterparty_bic", "—"),
                    "Account Code": ir.get("account_code", "—"),
                }
                for k, v in details.items():
                    st.write(f"**{k}:** {v}")


# ── PAGE 3 — P&L Attribution ──────────────────────────────────────────────────
def page_pl_attribution(recon: pd.DataFrame, pl_summary: pd.DataFrame):
    render_topbar()
    st.markdown('<div class="section-header">P&L Attribution by Exception Type</div>',
                unsafe_allow_html=True)

    exc = recon[recon["status"] == "EXCEPTION"].copy()
    exc["p_and_l_impact"] = pd.to_numeric(exc.get("p_and_l_impact", 0), errors="coerce").fillna(0)

    # P&L by type waterfall chart
    pnl_by_type = (exc.groupby("exception_code")["p_and_l_impact"]
                   .sum().sort_values(ascending=False).reset_index())
    pnl_by_type.columns = ["Code", "P&L"]

    fig_wf = go.Figure(go.Bar(
        x=pnl_by_type["Code"],
        y=pnl_by_type["P&L"],
        marker_color=[CYAN if v > 0 else RED for v in pnl_by_type["P&L"]],
        text=[f"£{v:,.0f}" for v in pnl_by_type["P&L"]],
        textfont_color="white",
        textposition="outside",
    ))
    fig_wf.update_layout(yaxis_title="P&L at Risk (£)", xaxis_title="", **PLOTLY_LAYOUT)
    fig_wf.update_yaxes(tickprefix="£", gridcolor=BORDER)
    st.plotly_chart(fig_wf, use_container_width=True)

    # Detailed breakdown table
    st.markdown('<div class="section-header">Cost Breakdown</div>', unsafe_allow_html=True)

    cost_cols = ["exception_code", "priority"]
    extra = [c for c in ("direct_pl_impact", "settlement_fail_cost",
                          "operational_cost", "total_cost") if c in exc.columns]
    if extra:
        grp = exc.groupby("exception_code")[extra].sum().reset_index()
        grp["count"] = exc.groupby("exception_code").size().values
        grp.insert(1, "count", grp.pop("count"))
        pri_map = exc.groupby("exception_code")["priority"].first()
        grp.insert(2, "priority", grp["exception_code"].map(pri_map))

        fmt = {c: "£{:,.0f}" for c in extra}
        styled_grp = (grp.style
                      .format(fmt)
                      .set_properties(**{"background-color": SURFACE,
                                         "color": TEXT, "border-color": BORDER}))
        st.dataframe(styled_grp, use_container_width=True)

    elif not pl_summary.empty:
        st.dataframe(pl_summary[pl_summary["exception_code"] != "TOTAL"]
                     .style.set_properties(**{"background-color": SURFACE, "color": TEXT}),
                     use_container_width=True)

    # Annual projection
    TRADING_DAYS = 252
    daily_total  = exc["p_and_l_impact"].sum()
    if "total_cost" in exc.columns:
        daily_total = exc["total_cost"].sum()
    annual = daily_total * TRADING_DAYS

    # Savings opportunity (85% reduction)
    savings = annual * 0.85

    st.markdown('<div class="section-header">Annual Cost Projection</div>',
                unsafe_allow_html=True)

    col_roi, col_sav = st.columns(2)
    with col_roi:
        st.markdown(f"""
        <div class="roi-box">
          <div class="roi-label">Projected Annual Cost</div>
          <div class="roi-amount">£{annual:,.0f}</div>
          <div style="color:{DIM};font-size:.72rem;margin-top:8px;">
            Based on {TRADING_DAYS} trading days × today's exception rate
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_sav:
        st.markdown(f"""
        <div class="savings-box">
          <div class="roi-label" style="color:{DIM};">AITX Savings Opportunity (85% reduction)</div>
          <div class="savings-amount">£{savings:,.0f}</div>
          <div style="color:{DIM};font-size:.72rem;margin-top:8px;">
            Estimated annual saving with AITX auto-resolution
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Priority pie + ops bar
    st.markdown('<div class="section-header">Cost by Priority Tier</div>',
                unsafe_allow_html=True)
    p1, p2 = st.columns(2)

    with p1:
        if "total_cost" in exc.columns:
            pri_cost = exc.groupby("priority")["total_cost"].sum().reset_index()
        else:
            pri_cost = exc.groupby("priority")["p_and_l_impact"].sum().reset_index()
            pri_cost.columns = ["priority", "total_cost"]
        fig_pri = px.pie(pri_cost, values="total_cost", names="priority",
                         color="priority",
                         color_discrete_map={"P1": RED, "P2": AMBER, "P3": GREEN},
                         hole=0.5)
        fig_pri.update_traces(textfont_color="white",
                              marker=dict(line=dict(color=BG, width=2)))
        st.plotly_chart(_fig(fig_pri), use_container_width=True)

    with p2:
        ac_col = "asset_class" if "asset_class" in exc.columns else None
        if ac_col:
            if "total_cost" in exc.columns:
                ac_cost = exc.groupby(ac_col)["total_cost"].sum().reset_index()
                ac_cost.columns = ["asset_class", "cost"]
            else:
                ac_cost = exc.groupby(ac_col)["p_and_l_impact"].sum().reset_index()
                ac_cost.columns = ["asset_class", "cost"]
            ac_cost = ac_cost.sort_values("cost", ascending=True)
            fig_ac = px.bar(ac_cost, x="cost", y="asset_class", orientation="h",
                            color="cost",
                            color_continuous_scale=[[0, BORDER], [1, CYAN]])
            fig_ac.update_layout(coloraxis_showscale=False,
                                 yaxis_title="", xaxis_title="Total Cost (£)", **PLOTLY_LAYOUT)
            fig_ac.update_xaxes(tickprefix="£", gridcolor=BORDER)
            st.plotly_chart(fig_ac, use_container_width=True)


# ── PAGE 4 — Upload Your Own Data ────────────────────────────────────────────
def page_upload():
    render_topbar()
    st.markdown('<div class="section-header">Upload Your Own Data — Client Demo Mode</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-panel">
      <div style="font-size:1.1rem;color:#8099AA;margin-bottom:8px;">
        Upload internal and counterparty trade files (CSV)
      </div>
      <div style="font-size:.78rem;color:#5A7080;">
        Expected columns: trade_id, isin, cusip, quantity, price, currency,
        settlement_date, counterparty_bic, account_code, trade_date, asset_class
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        internal_file    = st.file_uploader("Internal Trades CSV",    type="csv", key="up_int")
    with col_u2:
        counterparty_file = st.file_uploader("Counterparty Trades CSV", type="csv", key="up_cpty")

    if internal_file and counterparty_file:
        if st.button("Run Reconciliation", type="primary"):
            with st.spinner("Running AITX reconciliation engine…"):
                try:
                    # Write uploads to temp files
                    tmp_dir = tempfile.mkdtemp()
                    int_path  = os.path.join(tmp_dir, "internal_trades.csv")
                    cpty_path = os.path.join(tmp_dir, "counterparty_trades.csv")
                    out_path  = os.path.join(tmp_dir, "reconciliation_results.csv")

                    with open(int_path,  "wb") as f: f.write(internal_file.read())
                    with open(cpty_path, "wb") as f: f.write(counterparty_file.read())

                    # Run engine
                    sys.path.insert(0, BASE_DIR)
                    import reconciliation_engine as engine
                    engine.load_taxonomy(os.path.join(BASE_DIR, "exception_taxonomy.txt"))
                    results_df, summary = engine.run_reconciliation(
                        internal_path=int_path,
                        counterparty_path=cpty_path,
                        output_path=out_path,
                    )

                    st.success(f"Reconciliation complete — {len(results_df):,} records processed")

                    # Store in session state so other pages can use it
                    st.session_state["uploaded_recon"]    = results_df
                    st.session_state["uploaded_internal"] = pd.read_csv(
                        int_path, dtype={"trade_id": str, "counterparty_bic": str,
                                         "asset_class": str},
                        parse_dates=["trade_date", "settlement_date"])
                    st.session_state["upload_summary"] = summary

                    # Preview summary
                    n_exc = summary["total_exceptions"]
                    mr    = summary["match_rate"]
                    pl    = summary["total_pl_at_risk"]

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Match Rate",       f"{mr:.2f}%")
                    r2.metric("Total Exceptions", f"{n_exc:,}")
                    r3.metric("P&L at Risk",      f"£{pl:,.0f}")

                    st.markdown("**Navigate to Pages 1–3 to explore results.**")
                    st.dataframe(results_df[results_df["status"] == "EXCEPTION"]
                                 .head(50), use_container_width=True)

                except Exception as e:
                    st.error(f"Reconciliation failed: {e}")
                    st.exception(e)
    else:
        st.info("Upload both files above to run the reconciliation engine on your own data.")

    st.markdown("---")
    st.markdown(f"""
    <div style="color:{DIM};font-size:.72rem;">
      Demo files: use the synthetic CSVs in <code>data/</code> to test the upload flow.
      The engine supports any CSV with the required column schema.
    </div>
    """, unsafe_allow_html=True)


# ── navigation ────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown(f'<div style="font-size:1.1rem;font-weight:900;color:{CYAN};">'
                    f'⚡ AITX</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:.65rem;color:{DIM};margin-bottom:20px;">'
                    f'RECONCILIATION ENGINE</div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Executive Summary", "Exception Queue", "P&L Attribution", "Upload Data"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        using_upload = ("uploaded_recon" in st.session_state)
        if using_upload:
            st.markdown(f'<div style="color:{GREEN};font-size:.72rem;">'
                        f'● Using uploaded data</div>', unsafe_allow_html=True)
            if st.button("Reset to default data"):
                del st.session_state["uploaded_recon"]
                del st.session_state["uploaded_internal"]
                st.rerun()
        else:
            st.markdown(f'<div style="color:{DIM};font-size:.72rem;">'
                        f'Using synthetic demo data</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f'<div style="color:{DIM};font-size:.65rem;line-height:1.6;">'
                    f'Exception taxonomy: 10 codes<br>'
                    f'Model: XGBoost classifier<br>'
                    f'Priority: P1/P2/P3 tiers<br>'
                    f'P&L methodology: ISDA standard'
                    f'</div>', unsafe_allow_html=True)

    # ── load data (from upload or from disk) ──────────────────────────────────
    try:
        if "uploaded_recon" in st.session_state:
            recon    = st.session_state["uploaded_recon"]
            internal = st.session_state["uploaded_internal"]
        else:
            recon    = load_recon()
            internal = load_internal()

        # Enrich recon with asset_class if missing
        if "asset_class" not in recon.columns:
            slim = internal[["trade_id", "asset_class"]].drop_duplicates("trade_id")
            recon = recon.merge(slim, on="trade_id", how="left")

        try:
            pl_summary = load_pl_summary()
        except Exception:
            pl_summary = pd.DataFrame()

    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}\n\nRun `generate_synthetic_data.py` and "
                 f"`reconciliation_engine.py` first, or upload your own data.")
        st.stop()

    # ── route ─────────────────────────────────────────────────────────────────
    if page == "Executive Summary":
        page_executive(recon, pl_summary)
    elif page == "Exception Queue":
        page_exception_queue(recon, internal)
    elif page == "P&L Attribution":
        page_pl_attribution(recon, pl_summary)
    elif page == "Upload Data":
        page_upload()


if __name__ == "__main__":
    main()
