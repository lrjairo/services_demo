import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


# =============================================================================
# SETUP
# =============================================================================


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import dlt
    import duckdb
    from datetime import datetime, timedelta
    from scipy import stats
    return dlt, duckdb, go, make_subplots, np, pd, px, datetime, timedelta, stats


# =============================================================================
# TITLE & INTRODUCTION
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Marketing Analytics Platform

        **A comprehensive marketing analytics framework featuring descriptive, predictive, and prescriptive analytics**

        This notebook provides an end-to-end marketing analytics toolkit designed for modern
        multi-channel digital marketing operations. It demonstrates the full spectrum of
        analytical capabilities:

        1. **Descriptive Analytics** -- What happened? Historical performance metrics,
           trend analysis, and campaign comparisons across all advertising platforms.
        2. **Predictive Analytics** -- What will happen? Forecasting future spend and
           conversions, customer lifetime value prediction, and attribution modeling.
        3. **Prescriptive Analytics** -- What should we do? Budget optimization
           recommendations, channel allocation strategies, and actionable insights.

        ---

        ## Data Sources

        This platform integrates data from multiple advertising channels, simulating
        a realistic marketing data warehouse:

        | Platform | Data Type | Key Metrics |
        |----------|-----------|-------------|
        | **Meta Ads** (Facebook/Instagram) | Campaigns, Ad Sets, Ads | Impressions, Reach, CPM, CPC, Conversions |
        | **Google Ads** | Search, Display, Shopping | Clicks, CTR, CPC, Conversions, Quality Score |
        | **TikTok Ads** | Video Campaigns | Views, Engagement, CPV, Conversions |
        | **LinkedIn Ads** | B2B Campaigns | Impressions, Clicks, Leads, CPL |
        | **Twitter/X Ads** | Promoted Content | Impressions, Engagements, Followers |

        ---

        ## Analytical Framework

        ```
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                         MARKETING ANALYTICS                              │
        ├─────────────────────┬─────────────────────┬─────────────────────────────┤
        │   DESCRIPTIVE       │     PREDICTIVE      │      PRESCRIPTIVE           │
        │   "What happened?"  │   "What will        │   "What should we do?"      │
        │                     │    happen?"         │                             │
        ├─────────────────────┼─────────────────────┼─────────────────────────────┤
        │ • Revenue trends    │ • Spend forecasts   │ • Budget optimization       │
        │ • Channel mix       │ • Conversion        │ • Channel allocation        │
        │ • Campaign perf.    │   predictions       │ • Bid recommendations       │
        │ • Cohort analysis   │ • LTV modeling      │ • Campaign prioritization   │
        │ • Attribution       │ • Churn prediction  │ • Timing optimization       │
        └─────────────────────┴─────────────────────┴─────────────────────────────┘
        ```

        ---
        """
    )
    return


# =============================================================================
# BUSINESS QUESTIONS
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Essential Marketing Business Questions

        A data-driven marketing organization must answer questions across these
        critical domains. These questions drive every metric and dashboard that follows.

        ---

        ### Channel Performance & Efficiency
        - Which advertising channels deliver the highest ROAS (Return on Ad Spend)?
        - How does cost-per-acquisition (CPA) vary across channels and campaigns?
        - What is our blended customer acquisition cost, and is it sustainable?
        - Which channels are most efficient at each stage of the funnel?
        - How does creative performance vary across platforms?

        ### Budget & Resource Allocation
        - How should we allocate our monthly marketing budget across channels?
        - Which campaigns should receive increased investment vs. be paused?
        - What is the optimal frequency cap for each platform?
        - At what spend level do we see diminishing returns per channel?
        - How should budget shift seasonally based on historical performance?

        ### Customer Acquisition & Retention
        - What is our average customer lifetime value (LTV) by acquisition channel?
        - What is the payback period for customer acquisition by source?
        - Which audiences and demographics convert at the highest rates?
        - How does customer quality (retention, repeat purchase) vary by channel?
        - What is the optimal LTV:CAC ratio we should target?

        ### Campaign Effectiveness
        - Which campaigns are driving the most conversions at the lowest cost?
        - How does ad creative impact performance across platforms?
        - What is our conversion rate at each funnel stage by campaign?
        - How long does it take for campaigns to reach optimal performance?
        - What is the impact of seasonality on campaign effectiveness?

        ### Attribution & Incrementality
        - How much credit should each touchpoint receive for a conversion?
        - What is the true incremental value of each marketing channel?
        - How do upper-funnel activities impact lower-funnel conversion rates?
        - What is the cross-channel interaction effect between platforms?
        - Which attribution model best reflects our customer journey?

        ### Forecasting & Planning
        - What will our spend and conversions look like next month/quarter?
        - How much budget is needed to hit our acquisition targets?
        - What is the expected impact of increasing spend on a specific channel?
        - When should we launch campaigns for maximum effectiveness?
        - What are realistic targets based on historical trends?

        ---

        > **Best practice:** Marketing teams should review channel performance weekly,
        > conduct deep-dive campaign analysis bi-weekly, and reassess budget allocation
        > monthly. Forecasts and strategic planning should happen quarterly.
        """
    )
    return


# =============================================================================
# METRICS TREE
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Marketing Metrics Tree

        The following metrics tree defines every KPI used in this platform, organized
        hierarchically by domain. Each metric includes its definition, formula, and
        recommended benchmark ranges.

        ---

        ### Spend & Budget Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **Total Spend** | Total advertising expenditure | SUM(spend) | Based on budget |
        | **Spend by Channel** | Spend allocated to each platform | SUM(spend) WHERE channel = X | Varies by strategy |
        | **Spend Efficiency** | Revenue generated per dollar spent | Revenue / Spend | > 3:1 ROAS |
        | **Budget Utilization** | Actual spend vs. planned budget | Spend / Budget | 90-100% |
        | **Pacing** | Daily spend rate vs. target | Daily Spend / (Budget / Days) | 0.95-1.05 |

        ### Reach & Awareness Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **Impressions** | Total ad views | SUM(impressions) | Depends on budget |
        | **Reach** | Unique users who saw ads | SUM(reach) | Varies |
        | **Frequency** | Avg impressions per unique user | Impressions / Reach | 3-7 for awareness |
        | **CPM** | Cost per 1,000 impressions | (Spend / Impressions) x 1000 | $5-$25 varies by platform |
        | **Share of Voice** | Brand visibility vs. competitors | Brand Impressions / Total Market | Target 10%+ |

        ### Engagement Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **Clicks** | Total ad clicks | SUM(clicks) | -- |
        | **CTR** | Click-through rate | Clicks / Impressions | 0.5-2% display, 3-5% search |
        | **CPC** | Cost per click | Spend / Clicks | Varies by industry |
        | **Engagement Rate** | Interactions per impression | Engagements / Impressions | 1-3% |
        | **Video Views** | Video ad completions | SUM(video_views) | -- |
        | **View Rate** | Video completion rate | Video Views / Impressions | 15-30% |

        ### Conversion Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **Conversions** | Total conversion events | SUM(conversions) | Based on goals |
        | **CVR** | Conversion rate | Conversions / Clicks | 2-5% e-commerce |
        | **CPA** | Cost per acquisition | Spend / Conversions | Varies by product |
        | **ROAS** | Return on ad spend | Revenue / Spend | 3:1 to 5:1 |
        | **Revenue** | Total attributed revenue | SUM(revenue) | -- |
        | **AOV** | Average order value | Revenue / Conversions | Varies |

        ### Customer Value Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **CAC** | Customer acquisition cost | Spend / New Customers | Varies by industry |
        | **LTV** | Customer lifetime value | Avg Revenue x Retention Period | 3-5x CAC |
        | **LTV:CAC** | Ratio of value to cost | LTV / CAC | > 3:1 |
        | **Payback Period** | Months to recover CAC | CAC / Monthly Revenue | < 12 months |
        | **Cohort Retention** | % customers retained over time | Retained / Original Cohort | 30-40% Y1 |

        ---

        ### Key Metric Relationships

        | Driver Metric | Outcome Metric | Relationship |
        |--------------|----------------|--------------|
        | CTR | CPC | Higher CTR often reduces CPC through quality score |
        | Frequency | CVR | Optimal frequency improves CVR; over-frequency hurts it |
        | Creative Quality | CTR | Better creatives dramatically improve engagement |
        | Audience Targeting | CVR | Precise targeting improves conversion rates |
        | Landing Page Speed | CVR | Faster pages convert better |
        | Bid Strategy | CPA | Smart bidding can reduce CPA by 10-30% |
        | Budget Level | ROAS | Diminishing returns at high spend levels |
        | Seasonality | CPM | Competitive periods increase costs |

        ---
        """
    )
    return


# =============================================================================
# CONFIGURATION
# =============================================================================


@app.cell
def _():
    COMPANY_NAME = "Apex Digital Marketing"
    ANALYSIS_START = "2024-01-01"
    ANALYSIS_END = "2024-12-31"
    RANDOM_SEED = 42

    CHANNELS = [
        "Meta Ads",
        "Google Ads",
        "TikTok Ads",
        "LinkedIn Ads",
        "Twitter/X Ads",
    ]

    CHANNEL_CONFIG = {
        "Meta Ads": {
            "avg_cpm": 12.50,
            "avg_ctr": 0.012,
            "avg_cvr": 0.025,
            "budget_share": 0.35,
            "color": "#1877F2",
        },
        "Google Ads": {
            "avg_cpm": 8.00,
            "avg_ctr": 0.035,
            "avg_cvr": 0.040,
            "budget_share": 0.30,
            "color": "#4285F4",
        },
        "TikTok Ads": {
            "avg_cpm": 6.50,
            "avg_ctr": 0.008,
            "avg_cvr": 0.018,
            "budget_share": 0.15,
            "color": "#000000",
        },
        "LinkedIn Ads": {
            "avg_cpm": 35.00,
            "avg_ctr": 0.005,
            "avg_cvr": 0.032,
            "budget_share": 0.12,
            "color": "#0A66C2",
        },
        "Twitter/X Ads": {
            "avg_cpm": 7.00,
            "avg_ctr": 0.010,
            "avg_cvr": 0.015,
            "budget_share": 0.08,
            "color": "#1DA1F2",
        },
    }

    CAMPAIGN_TYPES = [
        "Brand Awareness",
        "Lead Generation",
        "Conversion",
        "Retargeting",
        "Prospecting",
    ]

    CAMPAIGN_TYPE_CONFIG = {
        "Brand Awareness": {"cvr_mult": 0.3, "cpm_mult": 0.8, "objective": "awareness"},
        "Lead Generation": {"cvr_mult": 0.8, "cpm_mult": 1.0, "objective": "leads"},
        "Conversion": {"cvr_mult": 1.5, "cpm_mult": 1.3, "objective": "purchase"},
        "Retargeting": {"cvr_mult": 2.5, "cpm_mult": 1.8, "objective": "purchase"},
        "Prospecting": {"cvr_mult": 0.6, "cpm_mult": 0.9, "objective": "traffic"},
    }

    # Seasonal multipliers (Jan=0, Dec=11)
    SEASONAL_WEIGHTS = {
        "Meta Ads": [0.9, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.95, 1.0, 1.1, 1.4, 1.5],
        "Google Ads": [0.95, 0.9, 0.95, 1.0, 1.0, 0.95, 0.9, 0.95, 1.0, 1.1, 1.3, 1.4],
        "TikTok Ads": [1.0, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.15, 1.1, 1.0, 1.1, 1.2],
        "LinkedIn Ads": [1.1, 1.0, 1.1, 1.1, 1.0, 0.85, 0.75, 0.8, 1.1, 1.15, 1.1, 0.9],
        "Twitter/X Ads": [0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.1, 1.15, 1.1],
    }

    COLORS = {
        "primary": "#0F4C81",
        "secondary": "#2E86AB",
        "accent": "#F57C20",
        "success": "#27AE60",
        "warning": "#F39C12",
        "danger": "#E74C3C",
        "info": "#3498DB",
        "muted": "#95A5A6",
    }

    CHANNEL_COLORS = {ch: cfg["color"] for ch, cfg in CHANNEL_CONFIG.items()}

    return (
        ANALYSIS_END,
        ANALYSIS_START,
        CAMPAIGN_TYPE_CONFIG,
        CAMPAIGN_TYPES,
        CHANNEL_COLORS,
        CHANNEL_CONFIG,
        CHANNELS,
        COLORS,
        COMPANY_NAME,
        RANDOM_SEED,
        SEASONAL_WEIGHTS,
    )


# =============================================================================
# CHART STYLE HELPER
# =============================================================================


@app.cell
def _(COLORS):
    def apply_chart_style(fig, title="", height=400):
        """Apply consistent styling to all Plotly figures."""
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#1a1a2e"), x=0.01),
            height=height,
            template="plotly_white",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#333"),
            margin=dict(t=60, b=40, l=50, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[
                COLORS["primary"],
                COLORS["accent"],
                COLORS["success"],
                COLORS["danger"],
                COLORS["secondary"],
                COLORS["warning"],
                COLORS["info"],
                COLORS["muted"],
            ],
        )
        return fig

    return (apply_chart_style,)


# =============================================================================
# DATA GENERATION
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Demo Company: Apex Digital Marketing

        Below we generate 12 months of realistic marketing data for **Apex Digital Marketing**,
        a multi-channel e-commerce company running campaigns across all major advertising
        platforms. The data includes:

        - Seasonal patterns (Q4 holiday peaks, summer slowdowns)
        - Platform-specific performance characteristics
        - Campaign type variations (awareness vs. conversion)
        - Realistic cost structures and conversion rates
        - Attribution data for multi-touch analysis

        ---
        """
    )
    return


@app.cell
def _(
    ANALYSIS_END,
    ANALYSIS_START,
    CAMPAIGN_TYPE_CONFIG,
    CAMPAIGN_TYPES,
    CHANNEL_CONFIG,
    CHANNELS,
    RANDOM_SEED,
    SEASONAL_WEIGHTS,
    np,
    pd,
):
    _rng = np.random.default_rng(RANDOM_SEED)

    # ---- Generate Campaigns ----
    _campaigns_list = []
    _campaign_id = 0

    for _channel in CHANNELS:
        for _ctype in CAMPAIGN_TYPES:
            for _i in range(3):  # 3 campaigns per channel-type combo
                _campaign_id += 1
                _campaigns_list.append({
                    "campaign_id": f"CMP{_campaign_id:04d}",
                    "campaign_name": f"{_channel.split()[0]} {_ctype} {_i+1}",
                    "channel": _channel,
                    "campaign_type": _ctype,
                    "start_date": pd.to_datetime(ANALYSIS_START) + pd.Timedelta(days=_rng.integers(0, 60)),
                    "daily_budget": _rng.uniform(100, 1000) * CHANNEL_CONFIG[_channel]["budget_share"],
                    "objective": CAMPAIGN_TYPE_CONFIG[_ctype]["objective"],
                    "status": _rng.choice(["Active", "Active", "Active", "Paused"], p=[0.7, 0.1, 0.1, 0.1]),
                })

    campaigns = pd.DataFrame(_campaigns_list)

    # ---- Generate Daily Ad Performance Data ----
    _dates = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="D")
    _monthly_budget = 150000  # Total monthly budget

    _daily_data = []
    _record_id = 0

    for _date in _dates:
        _month_idx = _date.month - 1
        _dow = _date.dayofweek
        _dow_factor = 1.0 if _dow < 5 else 0.75  # Lower weekend spend

        for _, _camp in campaigns.iterrows():
            if _date < _camp["start_date"]:
                continue
            if _camp["status"] == "Paused" and _rng.random() < 0.7:
                continue

            _channel = _camp["channel"]
            _ctype = _camp["campaign_type"]
            _cfg = CHANNEL_CONFIG[_channel]
            _type_cfg = CAMPAIGN_TYPE_CONFIG[_ctype]

            # Seasonal and day-of-week adjustments
            _seasonal = SEASONAL_WEIGHTS[_channel][_month_idx]
            _spend = max(0, _camp["daily_budget"] * _seasonal * _dow_factor * _rng.uniform(0.7, 1.3))

            # Calculate metrics based on spend
            _cpm = _cfg["avg_cpm"] * _type_cfg["cpm_mult"] * _rng.uniform(0.7, 1.4)
            _impressions = int((_spend / _cpm) * 1000) if _cpm > 0 else 0

            _ctr = _cfg["avg_ctr"] * _rng.uniform(0.6, 1.5)
            _clicks = int(_impressions * _ctr)

            _cvr = _cfg["avg_cvr"] * _type_cfg["cvr_mult"] * _rng.uniform(0.5, 1.8)
            _conversions = int(_clicks * _cvr)

            # Revenue (assuming $75-150 AOV)
            _aov = _rng.uniform(75, 150) if _type_cfg["objective"] == "purchase" else _rng.uniform(0, 25)
            _revenue = _conversions * _aov

            # Engagement metrics
            _engagements = int(_clicks * _rng.uniform(1.2, 2.5))
            _video_views = int(_impressions * _rng.uniform(0.15, 0.35)) if _channel == "TikTok Ads" else 0

            _record_id += 1
            _daily_data.append({
                "record_id": f"R{_record_id:06d}",
                "date": _date,
                "campaign_id": _camp["campaign_id"],
                "campaign_name": _camp["campaign_name"],
                "channel": _channel,
                "campaign_type": _ctype,
                "objective": _camp["objective"],
                "spend": round(_spend, 2),
                "impressions": _impressions,
                "reach": int(_impressions * _rng.uniform(0.6, 0.85)),
                "clicks": _clicks,
                "engagements": _engagements,
                "video_views": _video_views,
                "conversions": _conversions,
                "revenue": round(_revenue, 2),
            })

    daily_performance = pd.DataFrame(_daily_data)

    # ---- Generate Customer/Conversion Data ----
    _conversions_list = []
    _conv_id = 0

    for _, _row in daily_performance.iterrows():
        for _ in range(_row["conversions"]):
            _conv_id += 1
            _ltv_base = _rng.uniform(150, 800)
            _ltv_mult = {
                "Meta Ads": 1.0,
                "Google Ads": 1.2,
                "TikTok Ads": 0.8,
                "LinkedIn Ads": 1.5,
                "Twitter/X Ads": 0.7,
            }[_row["channel"]]

            _conversions_list.append({
                "conversion_id": f"CONV{_conv_id:06d}",
                "date": _row["date"],
                "campaign_id": _row["campaign_id"],
                "channel": _row["channel"],
                "campaign_type": _row["campaign_type"],
                "order_value": round(_rng.uniform(50, 250), 2),
                "customer_segment": _rng.choice(
                    ["New", "Returning", "Lapsed"],
                    p=[0.65, 0.25, 0.10]
                ),
                "device": _rng.choice(
                    ["Mobile", "Desktop", "Tablet"],
                    p=[0.55, 0.35, 0.10]
                ),
                "geo_region": _rng.choice(
                    ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
                    p=[0.20, 0.18, 0.22, 0.15, 0.25]
                ),
                "predicted_ltv": round(_ltv_base * _ltv_mult, 2),
                "first_touch_channel": _rng.choice(CHANNELS),
                "last_touch_channel": _row["channel"],
                "touchpoints": _rng.integers(1, 8),
            })

    conversions = pd.DataFrame(_conversions_list)

    # ---- Generate Attribution Data ----
    _attribution_list = []
    _attr_id = 0

    for _, _conv in conversions.iterrows():
        _n_touches = _conv["touchpoints"]
        _channels_touched = _rng.choice(CHANNELS, size=_n_touches, replace=True)
        _time_decay_weights = np.exp(-0.5 * np.arange(_n_touches)[::-1])
        _time_decay_weights = _time_decay_weights / _time_decay_weights.sum()

        for _i, _ch in enumerate(_channels_touched):
            _attr_id += 1
            _attribution_list.append({
                "attribution_id": f"ATTR{_attr_id:06d}",
                "conversion_id": _conv["conversion_id"],
                "date": _conv["date"],
                "touchpoint_order": _i + 1,
                "channel": _ch,
                "first_touch_credit": 1.0 if _i == 0 else 0.0,
                "last_touch_credit": 1.0 if _i == _n_touches - 1 else 0.0,
                "linear_credit": 1.0 / _n_touches,
                "time_decay_credit": _time_decay_weights[_i],
                "position_based_credit": (
                    0.4 if _i == 0 else
                    0.4 if _i == _n_touches - 1 else
                    0.2 / max(1, _n_touches - 2)
                ),
            })

    attribution = pd.DataFrame(_attribution_list)

    return attribution, campaigns, conversions, daily_performance


# =============================================================================
# DUCKDB SQL TRANSFORMATIONS - MART CONSTRUCTION
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Analytical Marts (SQL Transformations)

        The raw data is transformed into analytical marts using **SQL via DuckDB**.
        SQL is used wherever possible for maximum portability and readability.
        These marts are optimized for dashboarding and analysis.

        | Mart | Grain | Purpose |
        |------|-------|---------|
        | `mart_daily_channel` | Date x Channel | Daily channel performance metrics |
        | `mart_monthly_summary` | Month | Executive-level monthly KPI tracking |
        | `mart_campaign_performance` | Campaign | Campaign-level performance analysis |
        | `mart_channel_attribution` | Channel x Attribution Model | Multi-touch attribution by channel |
        | `mart_customer_cohort` | Month x Segment | Customer cohort analysis |
        | `mart_weekly_forecast` | Week | Weekly performance for forecasting |

        ---
        """
    )
    return


@app.cell
def _(attribution, campaigns, conversions, daily_performance, duckdb, pd):
    # Create DuckDB connection and register DataFrames
    _con = duckdb.connect(":memory:")
    _con.register("daily_performance", daily_performance)
    _con.register("campaigns", campaigns)
    _con.register("conversions", conversions)
    _con.register("attribution", attribution)

    # ---- mart_daily_channel: Daily channel performance ----
    mart_daily_channel = _con.execute("""
        SELECT
            date,
            channel,
            COUNT(DISTINCT campaign_id) as active_campaigns,
            SUM(spend) as spend,
            SUM(impressions) as impressions,
            SUM(reach) as reach,
            SUM(clicks) as clicks,
            SUM(engagements) as engagements,
            SUM(video_views) as video_views,
            SUM(conversions) as conversions,
            SUM(revenue) as revenue,
            -- Calculated metrics
            ROUND(SUM(spend) / NULLIF(SUM(impressions), 0) * 1000, 2) as cpm,
            ROUND(SUM(clicks)::FLOAT / NULLIF(SUM(impressions), 0) * 100, 2) as ctr,
            ROUND(SUM(spend) / NULLIF(SUM(clicks), 0), 2) as cpc,
            ROUND(SUM(conversions)::FLOAT / NULLIF(SUM(clicks), 0) * 100, 2) as cvr,
            ROUND(SUM(spend) / NULLIF(SUM(conversions), 0), 2) as cpa,
            ROUND(SUM(revenue) / NULLIF(SUM(spend), 0), 2) as roas,
            ROUND(SUM(impressions)::FLOAT / NULLIF(SUM(reach), 0), 2) as frequency
        FROM daily_performance
        GROUP BY date, channel
        ORDER BY date, channel
    """).fetchdf()

    # ---- mart_monthly_summary: Monthly executive summary ----
    mart_monthly_summary = _con.execute("""
        SELECT
            STRFTIME(date, '%Y-%m') as month_str,
            SUM(spend) as total_spend,
            SUM(impressions) as total_impressions,
            SUM(clicks) as total_clicks,
            SUM(conversions) as total_conversions,
            SUM(revenue) as total_revenue,
            COUNT(DISTINCT campaign_id) as active_campaigns,
            -- Blended metrics
            ROUND(SUM(spend) / NULLIF(SUM(impressions), 0) * 1000, 2) as blended_cpm,
            ROUND(SUM(clicks)::FLOAT / NULLIF(SUM(impressions), 0) * 100, 2) as blended_ctr,
            ROUND(SUM(spend) / NULLIF(SUM(clicks), 0), 2) as blended_cpc,
            ROUND(SUM(conversions)::FLOAT / NULLIF(SUM(clicks), 0) * 100, 2) as blended_cvr,
            ROUND(SUM(spend) / NULLIF(SUM(conversions), 0), 2) as blended_cpa,
            ROUND(SUM(revenue) / NULLIF(SUM(spend), 0), 2) as blended_roas
        FROM daily_performance
        GROUP BY STRFTIME(date, '%Y-%m')
        ORDER BY month_str
    """).fetchdf()

    # ---- mart_campaign_performance: Campaign-level analysis ----
    mart_campaign_performance = _con.execute("""
        SELECT
            dp.campaign_id,
            dp.campaign_name,
            dp.channel,
            dp.campaign_type,
            dp.objective,
            c.status,
            c.daily_budget,
            MIN(dp.date) as first_day,
            MAX(dp.date) as last_day,
            COUNT(DISTINCT dp.date) as active_days,
            SUM(dp.spend) as total_spend,
            SUM(dp.impressions) as total_impressions,
            SUM(dp.clicks) as total_clicks,
            SUM(dp.conversions) as total_conversions,
            SUM(dp.revenue) as total_revenue,
            ROUND(SUM(dp.spend) / NULLIF(SUM(dp.impressions), 0) * 1000, 2) as avg_cpm,
            ROUND(SUM(dp.clicks)::FLOAT / NULLIF(SUM(dp.impressions), 0) * 100, 2) as avg_ctr,
            ROUND(SUM(dp.spend) / NULLIF(SUM(dp.clicks), 0), 2) as avg_cpc,
            ROUND(SUM(dp.conversions)::FLOAT / NULLIF(SUM(dp.clicks), 0) * 100, 2) as avg_cvr,
            ROUND(SUM(dp.spend) / NULLIF(SUM(dp.conversions), 0), 2) as avg_cpa,
            ROUND(SUM(dp.revenue) / NULLIF(SUM(dp.spend), 0), 2) as roas
        FROM daily_performance dp
        JOIN campaigns c ON dp.campaign_id = c.campaign_id
        GROUP BY dp.campaign_id, dp.campaign_name, dp.channel, dp.campaign_type,
                 dp.objective, c.status, c.daily_budget
        ORDER BY total_spend DESC
    """).fetchdf()

    # ---- mart_channel_attribution: Attribution by channel ----
    mart_channel_attribution = _con.execute("""
        WITH conv_value AS (
            SELECT
                conversion_id,
                order_value
            FROM conversions
        )
        SELECT
            a.channel,
            COUNT(DISTINCT a.conversion_id) as total_conversions,
            SUM(cv.order_value) as total_revenue,
            ROUND(SUM(a.first_touch_credit), 2) as first_touch_conversions,
            ROUND(SUM(a.last_touch_credit), 2) as last_touch_conversions,
            ROUND(SUM(a.linear_credit), 2) as linear_conversions,
            ROUND(SUM(a.time_decay_credit), 2) as time_decay_conversions,
            ROUND(SUM(a.position_based_credit), 2) as position_based_conversions,
            ROUND(SUM(a.first_touch_credit * cv.order_value), 2) as first_touch_revenue,
            ROUND(SUM(a.last_touch_credit * cv.order_value), 2) as last_touch_revenue,
            ROUND(SUM(a.linear_credit * cv.order_value), 2) as linear_revenue,
            ROUND(SUM(a.time_decay_credit * cv.order_value), 2) as time_decay_revenue,
            ROUND(SUM(a.position_based_credit * cv.order_value), 2) as position_based_revenue
        FROM attribution a
        JOIN conv_value cv ON a.conversion_id = cv.conversion_id
        GROUP BY a.channel
        ORDER BY total_conversions DESC
    """).fetchdf()

    # ---- mart_customer_cohort: Customer cohort analysis ----
    mart_customer_cohort = _con.execute("""
        SELECT
            STRFTIME(date, '%Y-%m') as cohort_month,
            customer_segment,
            channel,
            COUNT(*) as conversions,
            SUM(order_value) as total_revenue,
            ROUND(AVG(order_value), 2) as avg_order_value,
            ROUND(AVG(predicted_ltv), 2) as avg_predicted_ltv,
            ROUND(AVG(touchpoints), 2) as avg_touchpoints
        FROM conversions
        GROUP BY STRFTIME(date, '%Y-%m'), customer_segment, channel
        ORDER BY cohort_month, customer_segment, channel
    """).fetchdf()

    # ---- mart_weekly_forecast: Weekly data for forecasting ----
    mart_weekly_forecast = _con.execute("""
        SELECT
            DATE_TRUNC('week', date) as week_start,
            SUM(spend) as spend,
            SUM(impressions) as impressions,
            SUM(clicks) as clicks,
            SUM(conversions) as conversions,
            SUM(revenue) as revenue,
            ROUND(SUM(revenue) / NULLIF(SUM(spend), 0), 2) as roas
        FROM daily_performance
        GROUP BY DATE_TRUNC('week', date)
        ORDER BY week_start
    """).fetchdf()

    # Close the connection
    _con.close()

    # Collect all marts for persistence
    marts_dict = {
        "mart_daily_channel": mart_daily_channel,
        "mart_monthly_summary": mart_monthly_summary,
        "mart_campaign_performance": mart_campaign_performance,
        "mart_channel_attribution": mart_channel_attribution,
        "mart_customer_cohort": mart_customer_cohort,
        "mart_weekly_forecast": mart_weekly_forecast,
    }

    # Also keep raw tables for reference
    raw_tables_dict = {
        "raw_daily_performance": daily_performance,
        "raw_campaigns": campaigns,
        "raw_conversions": conversions,
        "raw_attribution": attribution,
    }

    return (
        mart_campaign_performance,
        mart_channel_attribution,
        mart_customer_cohort,
        mart_daily_channel,
        mart_monthly_summary,
        mart_weekly_forecast,
        marts_dict,
        raw_tables_dict,
    )


# =============================================================================
# DLT PERSISTENCE TO DUCKDB
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Persistent Storage via dlt

        All analytical marts and raw tables are saved to a local DuckDB database using
        [dlt (data load tool)](https://dlthub.com/). This creates a queryable
        analytical warehouse that persists across notebook sessions.

        **Benefits:**
        - Query data with SQL outside the notebook
        - Connect to BI tools (Metabase, Superset, DBeaver)
        - Version and track data lineage
        - Easy integration with ETL pipelines

        ---
        """
    )
    return


@app.cell
def _(dlt, marts_dict, mo, pd, raw_tables_dict):
    _pipeline = dlt.pipeline(
        pipeline_name="marketing_analytics",
        destination="duckdb",
        dataset_name="marketing",
    )

    _load_results = []

    # Load marts
    for _name, _df in marts_dict.items():
        _clean = _df.copy()
        for _col in _clean.columns:
            if _clean[_col].dtype == "object":
                try:
                    _sample = _clean[_col].dropna().iloc[0] if len(_clean[_col].dropna()) > 0 else None
                    if _sample is not None and hasattr(_sample, "isoformat"):
                        _clean[_col] = pd.to_datetime(_clean[_col])
                except (IndexError, TypeError, ValueError):
                    pass

        _info = _pipeline.run(
            _clean.to_dict(orient="records"),
            table_name=_name,
            write_disposition="replace",
        )
        _load_results.append(f"| `{_name}` | {len(_df):,} rows | Mart |")

    # Load raw tables
    for _name, _df in raw_tables_dict.items():
        _clean = _df.copy()
        for _col in _clean.columns:
            if _clean[_col].dtype == "object":
                try:
                    _sample = _clean[_col].dropna().iloc[0] if len(_clean[_col].dropna()) > 0 else None
                    if _sample is not None and hasattr(_sample, "isoformat"):
                        _clean[_col] = pd.to_datetime(_clean[_col])
                except (IndexError, TypeError, ValueError):
                    pass

        _info = _pipeline.run(
            _clean.to_dict(orient="records"),
            table_name=_name,
            write_disposition="replace",
        )
        _load_results.append(f"| `{_name}` | {len(_df):,} rows | Raw |")

    _db_path = str(_pipeline.destination_client().config)

    dlt_load_info = mo.md(
        f"""
        **Database:** `{_db_path}`

        | Table | Size | Type |
        |-------|------|------|
        {"".join(chr(10) + r for r in _load_results)}

        All tables written with `write_disposition='replace'` -- re-running this cell
        refreshes the database with the latest generated data.

        **Query Example:**
        ```sql
        -- Connect to the DuckDB file and query:
        SELECT channel, SUM(spend) as total_spend, SUM(revenue) as total_revenue
        FROM marketing.mart_daily_channel
        GROUP BY channel
        ORDER BY total_spend DESC;
        ```
        """
    )
    dlt_load_info
    return (dlt_load_info,)


# =============================================================================
# DESCRIPTIVE ANALYTICS
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        # DESCRIPTIVE ANALYTICS

        ## What Happened?

        Descriptive analytics provides historical insights into marketing performance.
        These dashboards answer fundamental questions about past performance, trends,
        and patterns across channels, campaigns, and time periods.

        ---
        """
    )
    return


# =============================================================================
# DASHBOARD FILTERS
# =============================================================================


@app.cell
def _(CHANNELS, mo, campaigns):
    channel_filter = mo.ui.multiselect(
        options=CHANNELS,
        value=CHANNELS,
        label="Channels",
    )
    campaign_type_filter = mo.ui.multiselect(
        options=["Brand Awareness", "Lead Generation", "Conversion", "Retargeting", "Prospecting"],
        value=["Brand Awareness", "Lead Generation", "Conversion", "Retargeting", "Prospecting"],
        label="Campaign Types",
    )
    date_range_filter = mo.ui.dropdown(
        options={
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last 6 Months": 180,
            "Full Year": 365,
        },
        value="Full Year",
        label="Date Range",
    )
    campaign_selector = mo.ui.dropdown(
        options=dict(zip(campaigns["campaign_name"].tolist(), campaigns["campaign_id"].tolist())),
        value=campaigns["campaign_name"].iloc[0],
        label="Campaign Detail",
    )
    mo.hstack([channel_filter, campaign_type_filter, date_range_filter], justify="start", gap=1.5)
    return campaign_selector, campaign_type_filter, channel_filter, date_range_filter


# =============================================================================
# EXECUTIVE SUMMARY DASHBOARD
# =============================================================================


@app.cell
def _(
    CHANNEL_COLORS,
    apply_chart_style,
    go,
    mart_daily_channel,
    mart_monthly_summary,
    mo,
    channel_filter,
):
    # Filter by selected channels
    _daily = mart_daily_channel[mart_daily_channel["channel"].isin(channel_filter.value)]
    _monthly = mart_monthly_summary.copy()

    # Latest and prior month for KPI deltas
    _latest = _monthly.iloc[-1]
    _prev = _monthly.iloc[-2] if len(_monthly) > 1 else _latest

    # ---- KPI Cards ----
    def _kpi(title, val, prev_val, prefix="", suffix="", fmt=",.0f"):
        _f = go.Figure(
            go.Indicator(
                mode="number+delta",
                value=val,
                number={"prefix": prefix, "suffix": suffix, "valueformat": fmt, "font": {"size": 28}},
                delta={"reference": prev_val, "relative": True, "valueformat": ".1%"},
                title={"text": title, "font": {"size": 12, "color": "#555"}},
            )
        )
        _f.update_layout(
            height=110,
            margin=dict(t=40, b=10, l=15, r=15),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return _f

    _k1 = _kpi("Monthly Spend", _latest["total_spend"], _prev["total_spend"], prefix="$")
    _k2 = _kpi("Revenue", _latest["total_revenue"], _prev["total_revenue"], prefix="$")
    _k3 = _kpi("Conversions", _latest["total_conversions"], _prev["total_conversions"])
    _k4 = _kpi("ROAS", _latest["blended_roas"], _prev["blended_roas"], suffix="x", fmt=".2f")
    _k5 = _kpi("CPA", _latest["blended_cpa"], _prev["blended_cpa"], prefix="$", fmt=".2f")
    _k6 = _kpi("CTR", _latest["blended_ctr"], _prev["blended_ctr"], suffix="%", fmt=".2f")

    # ---- Spend & Revenue Trend ----
    _trend = go.Figure()
    _trend.add_trace(
        go.Scatter(
            x=_monthly["month_str"],
            y=_monthly["total_spend"],
            mode="lines+markers",
            name="Spend",
            line=dict(color="#E74C3C", width=3),
            marker=dict(size=7),
            hovertemplate="$%{y:,.0f}<extra>Spend</extra>",
        )
    )
    _trend.add_trace(
        go.Scatter(
            x=_monthly["month_str"],
            y=_monthly["total_revenue"],
            mode="lines+markers",
            name="Revenue",
            line=dict(color="#27AE60", width=3),
            marker=dict(size=7),
            hovertemplate="$%{y:,.0f}<extra>Revenue</extra>",
        )
    )
    apply_chart_style(_trend, "Monthly Spend vs. Revenue", height=350)
    _trend.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Channel Mix Donut ----
    _channel_totals = _daily.groupby("channel").agg({"spend": "sum"}).reset_index()
    _donut = go.Figure(
        go.Pie(
            labels=_channel_totals["channel"],
            values=_channel_totals["spend"],
            hole=0.5,
            marker=dict(colors=[CHANNEL_COLORS.get(c, "#999") for c in _channel_totals["channel"]]),
            textinfo="label+percent",
            hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
        )
    )
    apply_chart_style(_donut, "Spend by Channel (YTD)", height=350)

    # ---- ROAS Trend by Channel ----
    _roas_pivot = _daily.groupby(["date", "channel"]).agg(
        {"spend": "sum", "revenue": "sum"}
    ).reset_index()
    _roas_pivot["roas"] = _roas_pivot["revenue"] / _roas_pivot["spend"].replace(0, float("nan"))

    _roas_fig = go.Figure()
    for _ch in channel_filter.value:
        _ch_data = _roas_pivot[_roas_pivot["channel"] == _ch].copy()
        _ch_data["roas_ma"] = _ch_data["roas"].rolling(7, min_periods=1).mean()
        _roas_fig.add_trace(
            go.Scatter(
                x=_ch_data["date"],
                y=_ch_data["roas_ma"],
                mode="lines",
                name=_ch,
                line=dict(color=CHANNEL_COLORS.get(_ch, "#999"), width=2),
                hovertemplate="%{y:.2f}x<extra>%{fullData.name}</extra>",
            )
        )
    apply_chart_style(_roas_fig, "7-Day Rolling ROAS by Channel", height=350)
    _roas_fig.update_layout(
        yaxis_title="ROAS",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _roas_fig.add_hline(y=3, line_dash="dash", line_color="#999", annotation_text="3x Target")

    exec_summary_content = mo.vstack([
        mo.hstack([_k1, _k2, _k3, _k4, _k5, _k6], justify="space-around", widths="equal"),
        mo.hstack([_trend, _donut], widths="equal"),
        _roas_fig,
    ])
    return (exec_summary_content,)


# =============================================================================
# CHANNEL PERFORMANCE DASHBOARD
# =============================================================================


@app.cell
def _(
    CHANNEL_COLORS,
    COLORS,
    apply_chart_style,
    channel_filter,
    go,
    mart_daily_channel,
    mo,
    px,
):
    _daily = mart_daily_channel[mart_daily_channel["channel"].isin(channel_filter.value)]

    # ---- Channel Performance Comparison (Latest Month) ----
    _latest_month = _daily["date"].max().strftime("%Y-%m")
    _monthly_perf = _daily[_daily["date"].dt.strftime("%Y-%m") == _latest_month].groupby("channel").agg({
        "spend": "sum",
        "impressions": "sum",
        "clicks": "sum",
        "conversions": "sum",
        "revenue": "sum",
    }).reset_index()
    _monthly_perf["cpa"] = _monthly_perf["spend"] / _monthly_perf["conversions"].replace(0, float("nan"))
    _monthly_perf["roas"] = _monthly_perf["revenue"] / _monthly_perf["spend"].replace(0, float("nan"))

    _perf_bar = go.Figure()
    _perf_bar.add_trace(
        go.Bar(
            x=_monthly_perf["channel"],
            y=_monthly_perf["spend"],
            name="Spend",
            marker_color=COLORS["danger"],
            yaxis="y",
        )
    )
    _perf_bar.add_trace(
        go.Bar(
            x=_monthly_perf["channel"],
            y=_monthly_perf["revenue"],
            name="Revenue",
            marker_color=COLORS["success"],
            yaxis="y",
        )
    )
    _perf_bar.add_trace(
        go.Scatter(
            x=_monthly_perf["channel"],
            y=_monthly_perf["roas"],
            name="ROAS",
            mode="markers+text",
            marker=dict(size=16, color=COLORS["accent"], symbol="diamond"),
            text=[f"{v:.1f}x" for v in _monthly_perf["roas"]],
            textposition="top center",
            yaxis="y2",
        )
    )
    apply_chart_style(_perf_bar, f"Channel Performance ({_latest_month})", height=380)
    _perf_bar.update_layout(
        yaxis=dict(title="Amount ($)", tickprefix="$", tickformat=","),
        yaxis2=dict(title="ROAS", side="right", overlaying="y", range=[0, 8]),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- CPM/CPC/CPA by Channel ----
    _cost_metrics = _daily.groupby("channel").agg({
        "cpm": "mean",
        "cpc": "mean",
        "cpa": "mean",
    }).reset_index()

    _cost_fig = go.Figure()
    _cost_fig.add_trace(
        go.Bar(
            x=_cost_metrics["channel"],
            y=_cost_metrics["cpm"],
            name="CPM",
            marker_color=COLORS["primary"],
        )
    )
    _cost_fig.add_trace(
        go.Bar(
            x=_cost_metrics["channel"],
            y=_cost_metrics["cpc"],
            name="CPC",
            marker_color=COLORS["secondary"],
        )
    )
    _cost_fig.add_trace(
        go.Bar(
            x=_cost_metrics["channel"],
            y=_cost_metrics["cpa"],
            name="CPA",
            marker_color=COLORS["accent"],
        )
    )
    apply_chart_style(_cost_fig, "Average Cost Metrics by Channel", height=380)
    _cost_fig.update_layout(
        yaxis_tickprefix="$",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Impressions & CTR Trend ----
    _daily_agg = _daily.groupby("date").agg({
        "impressions": "sum",
        "clicks": "sum",
    }).reset_index()
    _daily_agg["ctr"] = (_daily_agg["clicks"] / _daily_agg["impressions"] * 100).round(2)

    _impr_fig = go.Figure()
    _impr_fig.add_trace(
        go.Bar(
            x=_daily_agg["date"],
            y=_daily_agg["impressions"],
            name="Impressions",
            marker_color=COLORS["muted"],
            opacity=0.6,
        )
    )
    _impr_fig.add_trace(
        go.Scatter(
            x=_daily_agg["date"],
            y=_daily_agg["ctr"],
            name="CTR %",
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            yaxis="y2",
        )
    )
    apply_chart_style(_impr_fig, "Daily Impressions & CTR", height=350)
    _impr_fig.update_layout(
        yaxis=dict(title="Impressions"),
        yaxis2=dict(title="CTR (%)", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Conversion Funnel ----
    _funnel_data = _daily.groupby("channel").agg({
        "impressions": "sum",
        "clicks": "sum",
        "conversions": "sum",
    }).reset_index()

    _funnel_figs = []
    for _ch in channel_filter.value[:3]:  # Top 3 channels
        _ch_data = _funnel_data[_funnel_data["channel"] == _ch].iloc[0] if len(_funnel_data[_funnel_data["channel"] == _ch]) > 0 else None
        if _ch_data is not None:
            _ff = go.Figure(go.Funnel(
                y=["Impressions", "Clicks", "Conversions"],
                x=[_ch_data["impressions"], _ch_data["clicks"], _ch_data["conversions"]],
                textinfo="value+percent initial",
                marker=dict(color=CHANNEL_COLORS.get(_ch, "#999")),
            ))
            apply_chart_style(_ff, f"{_ch} Funnel", height=280)
            _funnel_figs.append(_ff)

    channel_perf_content = mo.vstack([
        mo.hstack([_perf_bar, _cost_fig], widths="equal"),
        _impr_fig,
        mo.hstack(_funnel_figs, widths="equal") if _funnel_figs else mo.md("*Select channels to view funnels*"),
    ])
    return (channel_perf_content,)


# =============================================================================
# CAMPAIGN ANALYSIS DASHBOARD
# =============================================================================


@app.cell
def _(
    CHANNEL_COLORS,
    COLORS,
    apply_chart_style,
    campaign_type_filter,
    channel_filter,
    go,
    mart_campaign_performance,
    mo,
    px,
):
    _camps = mart_campaign_performance[
        (mart_campaign_performance["channel"].isin(channel_filter.value)) &
        (mart_campaign_performance["campaign_type"].isin(campaign_type_filter.value))
    ]

    # ---- Top Campaigns by Revenue ----
    _top_rev = _camps.nlargest(15, "total_revenue").sort_values("total_revenue")
    _top_rev_fig = px.bar(
        _top_rev,
        x="total_revenue",
        y="campaign_name",
        color="channel",
        orientation="h",
        color_discrete_map=CHANNEL_COLORS,
        hover_data={"total_spend": ":$,.0f", "roas": ":.2f", "total_conversions": ":,"},
    )
    apply_chart_style(_top_rev_fig, "Top 15 Campaigns by Revenue", height=450)
    _top_rev_fig.update_layout(
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Campaign Type Performance ----
    _type_perf = _camps.groupby("campaign_type").agg({
        "total_spend": "sum",
        "total_revenue": "sum",
        "total_conversions": "sum",
        "total_clicks": "sum",
    }).reset_index()
    _type_perf["roas"] = _type_perf["total_revenue"] / _type_perf["total_spend"].replace(0, float("nan"))
    _type_perf["cpa"] = _type_perf["total_spend"] / _type_perf["total_conversions"].replace(0, float("nan"))

    _type_fig = go.Figure()
    _type_fig.add_trace(
        go.Bar(
            x=_type_perf["campaign_type"],
            y=_type_perf["total_spend"],
            name="Spend",
            marker_color=COLORS["danger"],
        )
    )
    _type_fig.add_trace(
        go.Bar(
            x=_type_perf["campaign_type"],
            y=_type_perf["total_revenue"],
            name="Revenue",
            marker_color=COLORS["success"],
        )
    )
    apply_chart_style(_type_fig, "Spend vs. Revenue by Campaign Type", height=380)
    _type_fig.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- ROAS by Campaign Type ----
    _roas_type = go.Figure(
        go.Bar(
            x=_type_perf["campaign_type"],
            y=_type_perf["roas"],
            marker_color=[
                COLORS["success"] if v >= 3 else COLORS["warning"] if v >= 2 else COLORS["danger"]
                for v in _type_perf["roas"]
            ],
            text=[f"{v:.2f}x" for v in _type_perf["roas"]],
            textposition="outside",
        )
    )
    apply_chart_style(_roas_type, "ROAS by Campaign Type", height=380)
    _roas_type.add_hline(y=3, line_dash="dash", line_color="#999", annotation_text="3x Target")

    # ---- Scatter: Spend vs. ROAS ----
    _scatter = px.scatter(
        _camps,
        x="total_spend",
        y="roas",
        size="total_conversions",
        color="channel",
        hover_name="campaign_name",
        color_discrete_map=CHANNEL_COLORS,
        size_max=40,
    )
    apply_chart_style(_scatter, "Campaign Efficiency: Spend vs. ROAS", height=420)
    _scatter.update_layout(
        xaxis_title="Total Spend ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis_title="ROAS",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _scatter.add_hline(y=3, line_dash="dash", line_color="#999")

    campaign_analysis_content = mo.vstack([
        mo.hstack([_top_rev_fig, _type_fig], widths=[0.55, 0.45]),
        mo.hstack([_roas_type, _scatter], widths="equal"),
    ])
    return (campaign_analysis_content,)


# =============================================================================
# ATTRIBUTION DASHBOARD
# =============================================================================


@app.cell
def _(
    CHANNEL_COLORS,
    apply_chart_style,
    go,
    mart_channel_attribution,
    mo,
):
    _attr = mart_channel_attribution.copy()

    # ---- Attribution Model Comparison ----
    _models = ["first_touch", "last_touch", "linear", "time_decay", "position_based"]
    _model_labels = ["First Touch", "Last Touch", "Linear", "Time Decay", "Position-Based"]

    _attr_fig = go.Figure()
    for _i, (_model, _label) in enumerate(zip(_models, _model_labels)):
        _attr_fig.add_trace(
            go.Bar(
                x=_attr["channel"],
                y=_attr[f"{_model}_conversions"],
                name=_label,
            )
        )
    apply_chart_style(_attr_fig, "Attributed Conversions by Model", height=420)
    _attr_fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Revenue Attribution by Model ----
    _rev_attr = go.Figure()
    for _i, (_model, _label) in enumerate(zip(_models, _model_labels)):
        _rev_attr.add_trace(
            go.Bar(
                x=_attr["channel"],
                y=_attr[f"{_model}_revenue"],
                name=_label,
            )
        )
    apply_chart_style(_rev_attr, "Attributed Revenue by Model", height=420)
    _rev_attr.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- First vs. Last Touch Comparison ----
    _fl_compare = go.Figure()
    _fl_compare.add_trace(
        go.Bar(
            x=_attr["channel"],
            y=_attr["first_touch_conversions"],
            name="First Touch",
            marker_color=CHANNEL_COLORS.get("Meta Ads", "#1877F2"),
        )
    )
    _fl_compare.add_trace(
        go.Bar(
            x=_attr["channel"],
            y=_attr["last_touch_conversions"],
            name="Last Touch",
            marker_color=CHANNEL_COLORS.get("Google Ads", "#4285F4"),
        )
    )
    apply_chart_style(_fl_compare, "First Touch vs. Last Touch Attribution", height=380)
    _fl_compare.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Attribution Shift Analysis ----
    _attr["first_vs_last_diff"] = _attr["first_touch_conversions"] - _attr["last_touch_conversions"]
    _shift_fig = go.Figure(
        go.Bar(
            x=_attr["channel"],
            y=_attr["first_vs_last_diff"],
            marker_color=[
                "#27AE60" if v >= 0 else "#E74C3C" for v in _attr["first_vs_last_diff"]
            ],
            text=[f"{v:+.0f}" for v in _attr["first_vs_last_diff"]],
            textposition="outside",
        )
    )
    apply_chart_style(_shift_fig, "First Touch - Last Touch Difference (+ = Upper Funnel)", height=380)

    attribution_content = mo.vstack([
        mo.hstack([_attr_fig, _rev_attr], widths="equal"),
        mo.hstack([_fl_compare, _shift_fig], widths="equal"),
    ])
    return (attribution_content,)


# =============================================================================
# PREDICTIVE ANALYTICS
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        # PREDICTIVE ANALYTICS

        ## What Will Happen?

        Predictive analytics uses historical data to forecast future performance.
        This section includes spend forecasting, conversion predictions, and
        customer lifetime value modeling.

        ---
        """
    )
    return


@app.cell
def _(
    COLORS,
    apply_chart_style,
    go,
    mart_weekly_forecast,
    mo,
    np,
    pd,
    stats,
):
    _weekly = mart_weekly_forecast.copy()
    _weekly["week_num"] = range(len(_weekly))

    # ---- Simple Linear Regression Forecast ----
    # Fit trend on spend
    _x = _weekly["week_num"].values
    _y_spend = _weekly["spend"].values
    _y_conv = _weekly["conversions"].values
    _y_rev = _weekly["revenue"].values

    _slope_s, _intercept_s, _r_s, _, _ = stats.linregress(_x, _y_spend)
    _slope_c, _intercept_c, _r_c, _, _ = stats.linregress(_x, _y_conv)
    _slope_r, _intercept_r, _r_r, _, _ = stats.linregress(_x, _y_rev)

    # Forecast next 8 weeks
    _future_weeks = np.arange(len(_weekly), len(_weekly) + 8)
    _future_dates = [_weekly["week_start"].max() + pd.Timedelta(weeks=i+1) for i in range(8)]

    _forecast_spend = _intercept_s + _slope_s * _future_weeks
    _forecast_conv = _intercept_c + _slope_c * _future_weeks
    _forecast_rev = _intercept_r + _slope_r * _future_weeks

    # Add confidence intervals (simplified using standard error)
    _se_spend = np.std(_y_spend - (_intercept_s + _slope_s * _x))
    _se_conv = np.std(_y_conv - (_intercept_c + _slope_c * _x))
    _se_rev = np.std(_y_rev - (_intercept_r + _slope_r * _x))

    # ---- Spend Forecast Chart ----
    _spend_fig = go.Figure()
    _spend_fig.add_trace(
        go.Scatter(
            x=_weekly["week_start"],
            y=_weekly["spend"],
            mode="lines+markers",
            name="Actual Spend",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=5),
        )
    )
    _spend_fig.add_trace(
        go.Scatter(
            x=_future_dates,
            y=_forecast_spend,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
            marker=dict(size=7, symbol="star"),
        )
    )
    _spend_fig.add_trace(
        go.Scatter(
            x=_future_dates + _future_dates[::-1],
            y=list(_forecast_spend + 1.96*_se_spend) + list((_forecast_spend - 1.96*_se_spend)[::-1]),
            fill="toself",
            fillcolor="rgba(245, 124, 32, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=True,
        )
    )
    apply_chart_style(_spend_fig, f"Weekly Spend Forecast (R² = {_r_s**2:.2f})", height=380)
    _spend_fig.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Conversions Forecast Chart ----
    _conv_fig = go.Figure()
    _conv_fig.add_trace(
        go.Scatter(
            x=_weekly["week_start"],
            y=_weekly["conversions"],
            mode="lines+markers",
            name="Actual Conversions",
            line=dict(color=COLORS["success"], width=2),
            marker=dict(size=5),
        )
    )
    _conv_fig.add_trace(
        go.Scatter(
            x=_future_dates,
            y=_forecast_conv,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
            marker=dict(size=7, symbol="star"),
        )
    )
    _conv_fig.add_trace(
        go.Scatter(
            x=_future_dates + _future_dates[::-1],
            y=list(_forecast_conv + 1.96*_se_conv) + list(np.maximum(0, _forecast_conv - 1.96*_se_conv)[::-1]),
            fill="toself",
            fillcolor="rgba(39, 174, 96, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        )
    )
    apply_chart_style(_conv_fig, f"Weekly Conversions Forecast (R² = {_r_c**2:.2f})", height=380)
    _conv_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Revenue Forecast Chart ----
    _rev_fig = go.Figure()
    _rev_fig.add_trace(
        go.Scatter(
            x=_weekly["week_start"],
            y=_weekly["revenue"],
            mode="lines+markers",
            name="Actual Revenue",
            line=dict(color=COLORS["info"], width=2),
            marker=dict(size=5),
        )
    )
    _rev_fig.add_trace(
        go.Scatter(
            x=_future_dates,
            y=_forecast_rev,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
            marker=dict(size=7, symbol="star"),
        )
    )
    apply_chart_style(_rev_fig, f"Weekly Revenue Forecast (R² = {_r_r**2:.2f})", height=380)
    _rev_fig.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Forecast Summary ----
    _8wk_spend = sum(_forecast_spend)
    _8wk_conv = sum(_forecast_conv)
    _8wk_rev = sum(_forecast_rev)

    _forecast_summary = mo.md(
        f"""
        ### 8-Week Forecast Summary

        | Metric | Forecast | Weekly Avg | Trend |
        |--------|----------|------------|-------|
        | **Spend** | ${_8wk_spend:,.0f} | ${_8wk_spend/8:,.0f} | {'+' if _slope_s > 0 else ''}{_slope_s:,.0f}/week |
        | **Conversions** | {_8wk_conv:,.0f} | {_8wk_conv/8:,.0f} | {'+' if _slope_c > 0 else ''}{_slope_c:,.0f}/week |
        | **Revenue** | ${_8wk_rev:,.0f} | ${_8wk_rev/8:,.0f} | {'+' if _slope_r > 0 else ''}${_slope_r:,.0f}/week |
        | **Projected ROAS** | {_8wk_rev/_8wk_spend:.2f}x | -- | -- |
        """
    )

    forecast_content = mo.vstack([
        mo.hstack([_spend_fig, _conv_fig], widths="equal"),
        _rev_fig,
        _forecast_summary,
    ])
    return (forecast_content,)


# =============================================================================
# LTV PREDICTION
# =============================================================================


@app.cell
def _(
    CHANNEL_COLORS,
    COLORS,
    apply_chart_style,
    go,
    mart_customer_cohort,
    mo,
    px,
):
    _cohort = mart_customer_cohort.copy()

    # ---- LTV by Channel ----
    _ltv_channel = _cohort.groupby("channel").agg({
        "avg_predicted_ltv": "mean",
        "conversions": "sum",
        "total_revenue": "sum",
    }).reset_index()
    _ltv_channel = _ltv_channel.sort_values("avg_predicted_ltv", ascending=True)

    _ltv_fig = go.Figure(
        go.Bar(
            x=_ltv_channel["avg_predicted_ltv"],
            y=_ltv_channel["channel"],
            orientation="h",
            marker_color=[CHANNEL_COLORS.get(c, "#999") for c in _ltv_channel["channel"]],
            text=[f"${v:,.0f}" for v in _ltv_channel["avg_predicted_ltv"]],
            textposition="outside",
        )
    )
    apply_chart_style(_ltv_fig, "Predicted Customer LTV by Acquisition Channel", height=350)
    _ltv_fig.update_layout(xaxis_tickprefix="$", yaxis_title="")

    # ---- LTV by Segment ----
    _ltv_segment = _cohort.groupby("customer_segment").agg({
        "avg_predicted_ltv": "mean",
        "conversions": "sum",
    }).reset_index()

    _seg_fig = go.Figure(
        go.Bar(
            x=_ltv_segment["customer_segment"],
            y=_ltv_segment["avg_predicted_ltv"],
            marker_color=[COLORS["primary"], COLORS["success"], COLORS["warning"]],
            text=[f"${v:,.0f}" for v in _ltv_segment["avg_predicted_ltv"]],
            textposition="outside",
        )
    )
    apply_chart_style(_seg_fig, "Predicted LTV by Customer Segment", height=350)
    _seg_fig.update_layout(yaxis_tickprefix="$")

    # ---- LTV vs Volume Scatter ----
    _ltv_scatter = _cohort.groupby(["channel", "customer_segment"]).agg({
        "avg_predicted_ltv": "mean",
        "conversions": "sum",
        "avg_order_value": "mean",
    }).reset_index()

    _scatter_fig = px.scatter(
        _ltv_scatter,
        x="conversions",
        y="avg_predicted_ltv",
        size="avg_order_value",
        color="channel",
        hover_data={"customer_segment": True},
        color_discrete_map=CHANNEL_COLORS,
        size_max=50,
    )
    apply_chart_style(_scatter_fig, "LTV vs. Conversion Volume by Channel & Segment", height=400)
    _scatter_fig.update_layout(
        xaxis_title="Total Conversions",
        yaxis_title="Avg Predicted LTV ($)",
        yaxis_tickprefix="$",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Monthly LTV Trend ----
    _ltv_monthly = _cohort.groupby("cohort_month").agg({
        "avg_predicted_ltv": "mean",
        "conversions": "sum",
    }).reset_index()

    _ltv_trend = go.Figure()
    _ltv_trend.add_trace(
        go.Bar(
            x=_ltv_monthly["cohort_month"],
            y=_ltv_monthly["conversions"],
            name="Conversions",
            marker_color=COLORS["muted"],
            opacity=0.6,
        )
    )
    _ltv_trend.add_trace(
        go.Scatter(
            x=_ltv_monthly["cohort_month"],
            y=_ltv_monthly["avg_predicted_ltv"],
            name="Avg LTV",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    apply_chart_style(_ltv_trend, "Monthly Acquisition Volume & Average LTV", height=380)
    _ltv_trend.update_layout(
        yaxis=dict(title="Conversions"),
        yaxis2=dict(title="Avg LTV ($)", side="right", overlaying="y", tickprefix="$"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    ltv_content = mo.vstack([
        mo.hstack([_ltv_fig, _seg_fig], widths="equal"),
        mo.hstack([_scatter_fig, _ltv_trend], widths="equal"),
    ])
    return (ltv_content,)


# =============================================================================
# PRESCRIPTIVE ANALYTICS
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        # PRESCRIPTIVE ANALYTICS

        ## What Should We Do?

        Prescriptive analytics provides actionable recommendations based on data analysis.
        This section includes budget optimization, channel allocation recommendations,
        and strategic insights to improve marketing performance.

        ---
        """
    )
    return


@app.cell
def _(
    CHANNEL_COLORS,
    COLORS,
    apply_chart_style,
    go,
    mart_channel_attribution,
    mart_daily_channel,
    mart_monthly_summary,
    mo,
    np,
):
    # ---- Budget Optimization Analysis ----
    _daily = mart_daily_channel.copy()

    # Calculate efficiency metrics by channel
    _channel_eff = _daily.groupby("channel").agg({
        "spend": "sum",
        "revenue": "sum",
        "conversions": "sum",
        "clicks": "sum",
        "impressions": "sum",
    }).reset_index()

    _channel_eff["roas"] = _channel_eff["revenue"] / _channel_eff["spend"]
    _channel_eff["cpa"] = _channel_eff["spend"] / _channel_eff["conversions"]
    _channel_eff["current_share"] = _channel_eff["spend"] / _channel_eff["spend"].sum() * 100

    # Simple optimization: Allocate more to higher ROAS channels
    # Use ROAS-weighted allocation
    _total_roas = _channel_eff["roas"].sum()
    _channel_eff["roas_weight"] = _channel_eff["roas"] / _total_roas
    _channel_eff["optimal_share"] = (_channel_eff["roas_weight"] * 100).round(1)
    _channel_eff["share_change"] = _channel_eff["optimal_share"] - _channel_eff["current_share"]

    # ---- Current vs. Optimal Allocation ----
    _alloc_fig = go.Figure()
    _alloc_fig.add_trace(
        go.Bar(
            x=_channel_eff["channel"],
            y=_channel_eff["current_share"],
            name="Current Allocation",
            marker_color=COLORS["muted"],
        )
    )
    _alloc_fig.add_trace(
        go.Bar(
            x=_channel_eff["channel"],
            y=_channel_eff["optimal_share"],
            name="ROAS-Optimized",
            marker_color=COLORS["success"],
        )
    )
    apply_chart_style(_alloc_fig, "Current vs. ROAS-Optimized Budget Allocation", height=380)
    _alloc_fig.update_layout(
        yaxis_ticksuffix="%",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Recommended Changes ----
    _change_fig = go.Figure(
        go.Bar(
            x=_channel_eff["channel"],
            y=_channel_eff["share_change"],
            marker_color=[
                COLORS["success"] if v > 0 else COLORS["danger"]
                for v in _channel_eff["share_change"]
            ],
            text=[f"{v:+.1f}%" for v in _channel_eff["share_change"]],
            textposition="outside",
        )
    )
    apply_chart_style(_change_fig, "Recommended Budget Reallocation", height=380)
    _change_fig.update_layout(yaxis_ticksuffix="%")

    # ---- Efficiency Quadrant ----
    _avg_roas = _channel_eff["roas"].mean()
    _avg_spend = _channel_eff["spend"].mean()

    _quad_fig = go.Figure()
    for _, _row in _channel_eff.iterrows():
        _quad_fig.add_trace(
            go.Scatter(
                x=[_row["spend"]],
                y=[_row["roas"]],
                mode="markers+text",
                marker=dict(
                    size=30,
                    color=CHANNEL_COLORS.get(_row["channel"], "#999"),
                ),
                text=[_row["channel"].split()[0]],
                textposition="top center",
                name=_row["channel"],
                showlegend=False,
            )
        )
    _quad_fig.add_hline(y=_avg_roas, line_dash="dash", line_color="#999")
    _quad_fig.add_vline(x=_avg_spend, line_dash="dash", line_color="#999")
    _quad_fig.add_annotation(x=_avg_spend*1.5, y=_avg_roas*1.3, text="Scale Winners", showarrow=False, font=dict(size=11, color="#27AE60"))
    _quad_fig.add_annotation(x=_avg_spend*0.5, y=_avg_roas*1.3, text="Invest & Test", showarrow=False, font=dict(size=11, color="#3498DB"))
    _quad_fig.add_annotation(x=_avg_spend*1.5, y=_avg_roas*0.7, text="Optimize", showarrow=False, font=dict(size=11, color="#F39C12"))
    _quad_fig.add_annotation(x=_avg_spend*0.5, y=_avg_roas*0.7, text="Reduce/Cut", showarrow=False, font=dict(size=11, color="#E74C3C"))
    apply_chart_style(_quad_fig, "Channel Efficiency Quadrant (Spend vs. ROAS)", height=400)
    _quad_fig.update_layout(
        xaxis_title="Total Spend ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis_title="ROAS",
    )

    # ---- Actionable Recommendations ----
    _monthly = mart_monthly_summary.copy()
    _latest = _monthly.iloc[-1]

    # Generate recommendations
    _recs = []
    for _, _row in _channel_eff.iterrows():
        if _row["share_change"] > 5:
            _recs.append(f"**Increase {_row['channel']}** budget by {_row['share_change']:.0f}% -- ROAS of {_row['roas']:.2f}x exceeds portfolio average")
        elif _row["share_change"] < -5:
            _recs.append(f"**Reduce {_row['channel']}** budget by {abs(_row['share_change']):.0f}% -- ROAS of {_row['roas']:.2f}x underperforms")

    # Add general recommendations
    if _latest["blended_cpa"] > 50:
        _recs.append("**Review targeting** -- Blended CPA of ${:.0f} suggests audience refinement needed".format(_latest["blended_cpa"]))
    if _latest["blended_roas"] < 3:
        _recs.append("**Improve conversion rate** -- ROAS of {:.2f}x below 3x target; test landing pages".format(_latest["blended_roas"]))

    _rec_md = mo.md(
        "### Actionable Recommendations\n\n" +
        "\n".join([f"- {r}" for r in _recs]) +
        "\n\n---\n\n**Note:** These recommendations are based on ROAS optimization. "
        "Consider strategic factors (brand building, market expansion) that may justify "
        "maintaining investment in lower-ROAS channels."
    )

    optimization_content = mo.vstack([
        mo.hstack([_alloc_fig, _change_fig], widths="equal"),
        mo.hstack([_quad_fig, _rec_md], widths="equal"),
    ])
    return (optimization_content,)


# =============================================================================
# SCENARIO PLANNING
# =============================================================================


@app.cell
def _(COLORS, apply_chart_style, go, mart_monthly_summary, mo, np):
    # ---- Budget Scenario Simulator ----
    _monthly = mart_monthly_summary.copy()
    _current_spend = _monthly["total_spend"].mean()
    _current_roas = _monthly["blended_roas"].mean()
    _current_conv = _monthly["total_conversions"].mean()

    # Simulate different budget scenarios (with diminishing returns)
    _scenarios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    _scenario_data = []

    for _mult in _scenarios:
        _new_spend = _current_spend * _mult
        # Diminishing returns: ROAS decreases as spend increases
        _roas_adj = _current_roas * (1 - 0.1 * np.log(_mult) if _mult > 1 else 1 + 0.05 * np.log(1/_mult) if _mult < 1 else 1)
        _new_rev = _new_spend * _roas_adj
        _conv_adj = _current_conv * _mult * (1 - 0.08 * np.log(_mult) if _mult > 1 else 1)

        _scenario_data.append({
            "scenario": f"{int(_mult*100)}% Budget",
            "spend": _new_spend,
            "projected_revenue": _new_rev,
            "projected_roas": _roas_adj,
            "projected_conversions": _conv_adj,
            "multiplier": _mult,
        })

    _scenarios_df = mo.ui.table(_scenario_data, label="Budget Scenarios", page_size=10)

    # ---- Scenario Visualization ----
    _scen_fig = go.Figure()
    _scen_fig.add_trace(
        go.Scatter(
            x=[s["spend"] for s in _scenario_data],
            y=[s["projected_revenue"] for s in _scenario_data],
            mode="lines+markers",
            name="Projected Revenue",
            line=dict(color=COLORS["success"], width=3),
            marker=dict(size=10),
        )
    )
    _scen_fig.add_trace(
        go.Scatter(
            x=[s["spend"] for s in _scenario_data],
            y=[s["projected_roas"] * 50000 for s in _scenario_data],  # Scaled for visibility
            mode="lines+markers",
            name="ROAS (scaled)",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    apply_chart_style(_scen_fig, "Budget Scenario Analysis: Spend vs. Revenue & ROAS", height=400)
    _scen_fig.update_layout(
        xaxis_title="Monthly Spend ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis=dict(title="Revenue ($)", tickprefix="$", tickformat=","),
        yaxis2=dict(title="ROAS", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Mark current state
    _scen_fig.add_vline(x=_current_spend, line_dash="solid", line_color="#333", annotation_text="Current")

    # ---- ROI Curve ----
    _roi_fig = go.Figure()
    _spends = np.linspace(_current_spend * 0.3, _current_spend * 2.5, 50)
    _marginal_roas = [_current_roas * (1 - 0.15 * np.log(s/_current_spend)) if s > _current_spend else _current_roas * (1 + 0.08 * np.log(_current_spend/s)) for s in _spends]

    _roi_fig.add_trace(
        go.Scatter(
            x=_spends,
            y=_marginal_roas,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(39, 174, 96, 0.3)",
            line=dict(color=COLORS["success"], width=2),
            name="Marginal ROAS",
        )
    )
    _roi_fig.add_hline(y=3, line_dash="dash", line_color="#999", annotation_text="3x Target")
    _roi_fig.add_hline(y=1, line_dash="dash", line_color="#E74C3C", annotation_text="Break-even")
    apply_chart_style(_roi_fig, "Marginal ROAS Curve (Diminishing Returns)", height=380)
    _roi_fig.update_layout(
        xaxis_title="Monthly Spend ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis_title="Marginal ROAS",
    )

    scenario_content = mo.vstack([
        mo.md("### Budget Scenario Simulator"),
        mo.md("Explore how changes in budget affect projected revenue and ROAS, accounting for diminishing returns."),
        mo.hstack([_scen_fig, _roi_fig], widths="equal"),
        _scenarios_df,
    ])
    return (scenario_content,)


# =============================================================================
# DASHBOARD TABS
# =============================================================================


@app.cell
def _(
    attribution_content,
    campaign_analysis_content,
    channel_perf_content,
    exec_summary_content,
    forecast_content,
    ltv_content,
    mo,
    optimization_content,
    scenario_content,
):
    _tabs = mo.ui.tabs({
        "Executive Summary": exec_summary_content,
        "Channel Performance": channel_perf_content,
        "Campaign Analysis": campaign_analysis_content,
        "Attribution": attribution_content,
        "Forecasting": forecast_content,
        "Customer LTV": ltv_content,
        "Budget Optimization": optimization_content,
        "Scenario Planning": scenario_content,
    })
    _tabs
    return


# =============================================================================
# DATA EXPLORER
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Data Explorer

        Use the interactive table below to explore the analytical marts directly.
        Click column headers to sort, and use the search bar to filter.

        ---
        """
    )
    return


@app.cell
def _(marts_dict, mo, raw_tables_dict):
    _all_tables = {**marts_dict, **raw_tables_dict}
    table_selector = mo.ui.dropdown(
        options=list(_all_tables.keys()),
        value="mart_monthly_summary",
        label="Select Table",
    )
    table_selector
    return (table_selector,)


@app.cell
def _(marts_dict, mo, raw_tables_dict, table_selector):
    _all_tables = {**marts_dict, **raw_tables_dict}
    _selected_df = _all_tables[table_selector.value]
    mo.vstack([
        mo.md(
            f"**{table_selector.value}** -- {len(_selected_df):,} rows, "
            f"{len(_selected_df.columns)} columns"
        ),
        mo.ui.table(_selected_df, page_size=20, label=table_selector.value),
    ])
    return


# =============================================================================
# SQL QUERY INTERFACE
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## SQL Query Interface

        Write custom SQL queries against the marketing data marts. The following
        tables are available:

        - `mart_daily_channel` - Daily metrics by channel
        - `mart_monthly_summary` - Monthly executive KPIs
        - `mart_campaign_performance` - Campaign-level metrics
        - `mart_channel_attribution` - Multi-touch attribution
        - `mart_customer_cohort` - Cohort analysis
        - `mart_weekly_forecast` - Weekly data for forecasting

        ---
        """
    )
    return


@app.cell
def _(mo):
    sql_query = mo.ui.text_area(
        value="""SELECT
    channel,
    SUM(spend) as total_spend,
    SUM(revenue) as total_revenue,
    ROUND(SUM(revenue) / SUM(spend), 2) as roas,
    SUM(conversions) as total_conversions
FROM mart_daily_channel
GROUP BY channel
ORDER BY total_revenue DESC""",
        label="SQL Query",
        full_width=True,
        rows=8,
    )
    sql_query
    return (sql_query,)


@app.cell
def _(duckdb, marts_dict, mo, sql_query):
    try:
        _con = duckdb.connect(":memory:")
        for _name, _df in marts_dict.items():
            _con.register(_name, _df)

        _result = _con.execute(sql_query.value).fetchdf()
        _con.close()

        mo.vstack([
            mo.md(f"**Query Results:** {len(_result):,} rows"),
            mo.ui.table(_result, page_size=20, label="Query Results"),
        ])
    except Exception as e:
        mo.callout(f"**Query Error:** {str(e)}", kind="danger")
    return


# =============================================================================
# FOOTER
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        **Marketing Analytics Platform** | Built with
        [marimo](https://marimo.io), [Plotly](https://plotly.com/python/),
        [DuckDB](https://duckdb.org/), and [dlt](https://dlthub.com/)

        ---

        ## Analytics Methodology Summary

        ### Descriptive Analytics
        - Historical performance tracking across all channels
        - Campaign-level analysis with ROAS and CPA benchmarking
        - Multi-touch attribution using 5 different models
        - Funnel analysis from impressions to conversions

        ### Predictive Analytics
        - Linear regression-based forecasting with confidence intervals
        - Customer lifetime value prediction by channel and segment
        - Trend analysis and seasonality detection
        - R-squared goodness-of-fit metrics for forecast quality

        ### Prescriptive Analytics
        - ROAS-weighted budget optimization recommendations
        - Channel efficiency quadrant analysis
        - Scenario planning with diminishing returns modeling
        - Actionable recommendations based on data patterns

        ---

        > **Next steps for production deployment:**
        >
        > - Connect live API integrations (Meta Marketing API, Google Ads API, TikTok Ads API)
        > - Implement ML-based attribution (Markov chains, Shapley values)
        > - Add A/B test analysis and significance testing
        > - Build automated alerting for KPI anomalies
        > - Integrate with marketing automation platforms
        > - Add cohort analysis with retention curves
        """
    )
    return


if __name__ == "__main__":
    app.run()
