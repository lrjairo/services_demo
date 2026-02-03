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
    return dlt, duckdb, go, make_subplots, np, pd, px


# =============================================================================
# TITLE & INTRODUCTION
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Field Services Analytics Platform

        **A comprehensive analytics framework for field services businesses**

        This notebook provides a complete analytics toolkit designed for field services
        companies -- HVAC, plumbing, electrical, appliance repair, and general maintenance.
        It is structured around four pillars:

        1. **Essential Business Questions** -- The critical questions every field services
           operator must answer to run a profitable, efficient, customer-centric operation.
        2. **Metrics Tree** -- A hierarchical KPI framework with precise definitions,
           formulas, and associations that map directly to the business questions.
        3. **Demo Company Data Marts** -- Realistic synthetic data for *Summit Field Services*,
           a multi-trade home services company, built into analytical marts.
        4. **Persistent Storage** -- All marts are saved to a local DuckDB database via
           [dlt](https://dlthub.com/) for downstream querying and BI tool integration.
        5. **Interactive Dashboards** -- Five Plotly-powered dashboards covering executive
           summary, revenue, technician performance, customer health, and operations.

        ---

        **Industry context:** Field services businesses face unique challenges -- mobile
        workforces, unpredictable demand, high customer expectations for responsiveness, and
        thin margins on labor-intensive work. The metrics and dashboards here are tailored to
        these realities, drawing on best practices from ServiceTitan, Housecall Pro, and
        industry benchmarks from the Service Council and ACCA.

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
        ## 1. Essential Business Questions

        Running a successful field services business requires answering questions across
        six critical domains. These questions drive every metric and dashboard that follows.

        ---

        ### Revenue & Profitability
        - Are we growing revenue month-over-month and year-over-year?
        - What is our gross margin by service line, and which services are most profitable?
        - What is our average revenue per job, and how does it vary by service type and customer segment?
        - How much revenue does each technician generate?
        - What percentage of revenue comes from recurring service agreements vs. one-time jobs?
        - Are we pricing our services competitively while maintaining target margins?

        ### Operational Efficiency
        - What is our first-time fix rate, and which service types or technicians lag behind?
        - How much time do technicians spend on productive wrench time vs. travel?
        - What is our average job completion time by service type?
        - How many jobs per day does each technician complete?
        - What is our callback/rework rate, and what is it costing us in margin and reputation?

        ### Customer Health
        - What is our average customer satisfaction score, and is it trending up or down?
        - What is our customer retention rate -- how many customers return for repeat business?
        - What is the lifetime value of our customers by segment (residential vs. commercial)?
        - Which customers are at highest risk of churning?
        - How does service agreement penetration affect retention and lifetime value?

        ### Workforce Performance
        - What is our technician utilization rate (billable hours vs. available hours)?
        - How does performance vary across technicians in speed, quality, and revenue generation?
        - Are our highest-skilled technicians being dispatched to the highest-value jobs?
        - What is the revenue and margin impact of technician skill levels?

        ### Scheduling & Dispatch
        - What is our on-time arrival rate, and how does it affect customer satisfaction?
        - What is the average time from job request to technician on-site?
        - How well do we adhere to the daily dispatch schedule?
        - What are our peak demand periods, and are we staffed appropriately?

        ### Sales & Growth
        - What is our quote-to-close conversion rate?
        - How long does it take to convert a lead to a completed job?
        - What is our service agreement renewal rate?
        - Which customer segments and zip codes have the highest growth potential?

        ---

        > **Best practice:** Leading field services companies review these questions in a
        > weekly operations meeting, monthly business review, and quarterly strategic planning
        > session. The dashboards below are designed to support all three cadences.
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
        ## 2. Metrics Tree

        The following metrics tree defines every KPI used in this platform, organized
        hierarchically by domain. Each metric includes its definition, formula, and
        recommended benchmark range for field services businesses.

        ---

        ### Revenue Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **Total Revenue** | Gross revenue from all completed jobs | SUM(job_revenue) | Growing 10-20% YoY |
        | **Avg Revenue Per Job** | Average ticket size per completed job | Total Revenue / Completed Jobs | $250-$500 (residential) |
        | **Revenue Per Technician** | Revenue attributed to each active tech | Total Revenue / Active Technicians | $150K-$250K/yr |
        | **Agreement Revenue %** | Share of revenue from service agreements | Agreement Revenue / Total Revenue | Target 20-35% |
        | **Revenue Per Customer** | Average revenue per unique customer | Total Revenue / Unique Customers | Varies by segment |

        ### Profitability Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **Gross Profit** | Revenue minus direct job costs | Revenue - (Labor + Parts + Travel Cost) | -- |
        | **Gross Margin %** | Percentage of revenue retained after direct costs | Gross Profit / Revenue | 45-55% |
        | **Avg Job Margin** | Average gross margin per completed job | AVG(Job Profit / Job Revenue) | 40-50% |
        | **Cost Per Job** | Average direct cost to complete a job | Total Direct Costs / Completed Jobs | Monitor trend |
        | **Parts Cost Ratio** | Parts cost as share of revenue | Parts Cost / Revenue | 20-35% |

        ### Efficiency Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **First-Time Fix Rate** | Jobs resolved on the first visit | First-Visit Fixes / Total Jobs | 75-90% |
        | **Callback Rate** | Jobs requiring a return visit | Callbacks / Total Jobs | Below 10% |
        | **Avg Job Duration** | Average on-site time per job (hours) | AVG(duration_hours) | Varies by trade |
        | **Wrench Time %** | Productive on-site hours vs. total hours | On-Site Hours / Total Hours | 60-70% |
        | **Travel Time %** | Travel hours vs. total hours | Travel Hours / Total Hours | Below 20% |
        | **Jobs Per Tech Per Day** | Average daily jobs completed per tech | Completed Jobs / Tech Work Days | 3-6 |
        | **Technician Utilization** | Billable hours vs. available hours | Billable Hours / Available Hours | 70-85% |

        ### Customer Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **CSAT Score** | Average satisfaction rating (1-5 scale) | AVG(csat_rating) | 4.2+ |
        | **Net Promoter Score** | Customer loyalty indicator | % Promoters - % Detractors | 50+ |
        | **Customer Retention Rate** | Customers with repeat business | Repeat Customers / Total Customers | 60-75% |
        | **Customer Lifetime Value** | Estimated total revenue per customer | Avg Revenue/Customer x Avg Tenure | Maximize |
        | **Agreement Penetration** | Customers on service agreements | Agreement Customers / Total | 20-30% |

        ### Scheduling Metrics
        | Metric | Definition | Formula | Benchmark |
        |--------|-----------|---------|-----------|
        | **On-Time Arrival Rate** | Arrivals within the scheduled window | On-Time Arrivals / Total Jobs | 85-95% |
        | **Avg Response Time** | Hours from request to technician arrival | AVG(arrival - created) | Same-day for urgent |
        | **Schedule Adherence** | Jobs completed on the original date | On-Schedule / Total Scheduled | 80-90% |
        | **Same-Day Service Rate** | Jobs completed day-of request | Same-Day Jobs / Total Jobs | 30-50% |

        ---

        ### Key Metric Associations

        These are the critical cause-and-effect relationships between metrics:

        | Driver Metric | Outcome Metric | Relationship |
        |--------------|----------------|--------------|
        | First-Time Fix Rate | Callback Rate | Higher FTFR directly reduces costly callbacks |
        | Technician Utilization | Revenue Per Tech | More billable hours drive higher per-tech revenue |
        | CSAT Score | Customer Retention | Satisfaction is the #1 predictor of repeat business |
        | Agreement Penetration | Revenue Stability | Agreements reduce revenue volatility by 30-40% |
        | On-Time Arrival | CSAT Score | Punctuality is the top driver of satisfaction |
        | Avg Job Duration | Jobs Per Tech/Day | Faster completions enable higher daily throughput |
        | Skill Level | First-Time Fix Rate | Higher-skilled techs fix more on the first visit |
        | Travel Time % | Utilization | Excessive travel directly erodes productive utilization |
        | Callback Rate | Gross Margin | Each callback costs $150-$300 in unbillable labor |
        | Revenue Per Job | Gross Margin | Higher ticket sizes typically carry better margins |

        ---

        > **Best practice:** Display the top 5-8 metrics on a wall-mounted TV in the dispatch
        > center. Technicians and dispatchers should see FTFR, on-time rate, jobs completed
        > today, and CSAT in real time. Reserve the full metrics tree for management reviews.
        """
    )
    return


# =============================================================================
# CONFIGURATION
# =============================================================================


@app.cell
def _():
    COMPANY_NAME = "Summit Field Services"
    ANALYSIS_START = "2025-01-01"
    ANALYSIS_END = "2025-12-31"
    RANDOM_SEED = 42

    SERVICE_TYPES = [
        "HVAC",
        "Plumbing",
        "Electrical",
        "Appliance Repair",
        "General Maintenance",
    ]

    SERVICE_CONFIG = {
        "HVAC": {
            "avg_revenue": 420,
            "revenue_std": 180,
            "avg_duration": 2.5,
            "duration_std": 1.0,
            "parts_pct": 0.35,
        },
        "Plumbing": {
            "avg_revenue": 320,
            "revenue_std": 140,
            "avg_duration": 2.0,
            "duration_std": 0.8,
            "parts_pct": 0.30,
        },
        "Electrical": {
            "avg_revenue": 280,
            "revenue_std": 120,
            "avg_duration": 2.2,
            "duration_std": 0.9,
            "parts_pct": 0.25,
        },
        "Appliance Repair": {
            "avg_revenue": 200,
            "revenue_std": 80,
            "avg_duration": 1.5,
            "duration_std": 0.5,
            "parts_pct": 0.40,
        },
        "General Maintenance": {
            "avg_revenue": 150,
            "revenue_std": 50,
            "avg_duration": 1.2,
            "duration_std": 0.4,
            "parts_pct": 0.20,
        },
    }

    # Monthly seasonal multipliers (Jan=index 0 ... Dec=index 11)
    SEASONAL_WEIGHTS = {
        "HVAC": [0.9, 0.8, 0.6, 0.5, 0.7, 1.3, 1.5, 1.5, 1.2, 0.7, 0.8, 1.0],
        "Plumbing": [1.2, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.1, 1.2],
        "Electrical": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Appliance Repair": [1.0, 1.0, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 1.1, 1.1, 1.0],
        "General Maintenance": [0.7, 0.7, 1.2, 1.3, 1.2, 0.9, 0.8, 0.8, 1.1, 1.3, 1.0, 0.7],
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

    SERVICE_COLORS = {
        "HVAC": "#2E86AB",
        "Plumbing": "#27AE60",
        "Electrical": "#F57C20",
        "Appliance Repair": "#E74C3C",
        "General Maintenance": "#9B59B6",
    }

    return (
        ANALYSIS_END,
        ANALYSIS_START,
        COLORS,
        COMPANY_NAME,
        RANDOM_SEED,
        SERVICE_COLORS,
        SERVICE_CONFIG,
        SERVICE_TYPES,
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
        ## 3. Demo Company: Summit Field Services

        Below we generate 12 months of realistic operational data for **Summit Field Services**,
        a multi-trade home services company operating in the Denver metro area. The data
        includes seasonal patterns (HVAC peaks in summer/winter), technician skill variation,
        commercial vs. residential customer segments, and realistic cost structures.

        ---
        """
    )
    return


@app.cell
def _(
    ANALYSIS_END,
    ANALYSIS_START,
    RANDOM_SEED,
    SERVICE_CONFIG,
    SERVICE_TYPES,
    SEASONAL_WEIGHTS,
    np,
    pd,
):
    _rng = np.random.default_rng(RANDOM_SEED)

    # ---- Technicians ----
    _tech_names = [
        "Alex Rivera",
        "Jordan Chen",
        "Taylor Brooks",
        "Morgan Patel",
        "Casey Sullivan",
        "Drew Martinez",
        "Riley Thompson",
        "Quinn Foster",
        "Avery Washington",
        "Blake Henderson",
        "Dakota Nguyen",
        "Emerson Clark",
        "Finley Cooper",
        "Hayden Bell",
        "Jamie Ross",
    ]
    _tech_specialties = [
        "HVAC", "HVAC", "HVAC",
        "Plumbing", "Plumbing", "Plumbing",
        "Electrical", "Electrical",
        "Appliance Repair", "Appliance Repair",
        "General Maintenance", "General Maintenance",
        "HVAC", "Plumbing", "Electrical",
    ]
    _tech_hire_dates = pd.to_datetime([
        "2020-03-15", "2021-06-01", "2019-01-10", "2022-02-20", "2020-08-05",
        "2021-11-15", "2023-01-08", "2019-07-22", "2022-09-01", "2021-04-10",
        "2023-06-15", "2020-12-01", "2024-03-01", "2024-06-15", "2022-05-20",
    ])

    technicians = pd.DataFrame({
        "tech_id": [f"T{i:03d}" for i in range(1, 16)],
        "tech_name": _tech_names,
        "specialty": _tech_specialties,
        "hire_date": _tech_hire_dates,
        "hourly_rate": [45, 42, 48, 40, 43, 38, 35, 47, 36, 39, 33, 44, 32, 30, 41],
        "skill_level": [5, 4, 5, 4, 4, 3, 3, 5, 3, 4, 2, 4, 2, 2, 4],
    })

    # ---- Customers ----
    _n_customers = 400
    _first = [
        "James", "Maria", "Robert", "Linda", "Michael", "Barbara", "William",
        "Elizabeth", "David", "Jennifer", "Richard", "Patricia", "Joseph", "Susan",
        "Thomas", "Jessica", "Christopher", "Sarah", "Charles", "Karen", "Daniel",
        "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Dorothy",
        "Donald", "Sandra",
    ]
    _last = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        "Ramirez", "Lewis", "Robinson",
    ]
    _cust_names = [
        f"{_first[i % len(_first)]} {_last[(i * 7 + 3) % len(_last)]}"
        for i in range(_n_customers)
    ]
    _cust_types = _rng.choice(
        ["Residential", "Commercial"], size=_n_customers, p=[0.72, 0.28]
    )
    _join_dates = (
        pd.to_datetime(ANALYSIS_START)
        - pd.to_timedelta(_rng.integers(30, 1200, size=_n_customers), unit="D")
    )
    _has_agreement = _rng.random(_n_customers) < 0.22
    _agreement_vals = np.where(
        _has_agreement,
        np.where(
            _cust_types == "Residential",
            _rng.uniform(29, 79, _n_customers),
            _rng.uniform(99, 299, _n_customers),
        ),
        0,
    )

    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, _n_customers + 1)],
        "customer_name": _cust_names,
        "customer_type": _cust_types,
        "zip_code": _rng.choice(
            ["80301", "80302", "80303", "80304", "80305", "80401", "80501", "80516"],
            _n_customers,
        ),
        "join_date": _join_dates,
        "has_service_agreement": _has_agreement,
        "agreement_monthly_value": np.round(_agreement_vals, 2),
    })

    # ---- Jobs / Work Orders ----
    _dates = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="D")
    _base_daily = 10

    _jobs_list = []
    _job_id = 0

    for _date in _dates:
        _month_idx = _date.month - 1
        _dow = _date.dayofweek
        _dow_factor = 1.0 if _dow < 5 else 0.3
        _growth = 1.0 + (_date.month - 1) * 0.015

        for _svc in SERVICE_TYPES:
            _seasonal = SEASONAL_WEIGHTS[_svc][_month_idx]
            _expected = _base_daily / len(SERVICE_TYPES) * _seasonal * _dow_factor * _growth
            _n_jobs = _rng.poisson(max(_expected, 0.1))
            _cfg = SERVICE_CONFIG[_svc]

            for _ in range(_n_jobs):
                _job_id += 1

                # Customer selection
                _ci = _rng.integers(0, _n_customers)
                _cust = customers.iloc[_ci]

                # Technician selection (prefer specialists)
                _spec_mask = technicians["specialty"] == _svc
                if _spec_mask.any() and _rng.random() < 0.85:
                    _ti = _rng.choice(technicians[_spec_mask].index)
                else:
                    _ti = _rng.choice(technicians.index)
                _tech = technicians.iloc[_ti]

                # Job type
                _job_type = _rng.choice(
                    ["Repair", "Installation", "Maintenance", "Inspection"],
                    p=[0.50, 0.15, 0.25, 0.10],
                )

                # Revenue
                _comm_mult = 1.4 if _cust["customer_type"] == "Commercial" else 1.0
                _inst_mult = 2.5 if _job_type == "Installation" else 1.0
                _revenue = max(
                    50.0,
                    _rng.normal(
                        _cfg["avg_revenue"] * _comm_mult * _inst_mult,
                        _cfg["revenue_std"],
                    ),
                )

                # Costs
                _parts_cost = _revenue * _cfg["parts_pct"] * _rng.uniform(0.7, 1.3)
                _duration = max(
                    0.5,
                    _rng.normal(
                        _cfg["avg_duration"] * (1.5 if _job_type == "Installation" else 1.0),
                        _cfg["duration_std"],
                    ),
                )
                _travel = max(0.15, _rng.exponential(0.5))
                _labor_cost = _tech["hourly_rate"] * _duration

                # Quality (influenced by tech skill)
                _skill = _tech["skill_level"]
                _ftfr_prob = 0.60 + _skill * 0.07
                _ftf = bool(_rng.random() < _ftfr_prob)
                _callback = (not _ftf) and bool(_rng.random() < 0.4)

                # On-time arrival
                _on_time = bool(_rng.random() < (0.70 + _skill * 0.05))

                # CSAT (correlated with skill, fix success, punctuality)
                _base_csat = 3.0 + _skill * 0.3
                _csat_adj = (0.3 if _ftf else -0.3) + (0.2 if _on_time else -0.4)
                _csat = float(min(5.0, max(1.0, _rng.normal(_base_csat + _csat_adj, 0.5))))

                # Scheduling
                _lead = int(_rng.choice([0, 0, 0, 1, 1, 2, 3, 5, 7]))
                _created = _date - pd.Timedelta(days=_lead)

                # Status (95% completed)
                _status = "Completed" if _rng.random() < 0.95 else "Cancelled"

                _gross_profit = (
                    _revenue - _parts_cost - _labor_cost - _travel * _tech["hourly_rate"]
                    if _status == "Completed"
                    else 0.0
                )

                _jobs_list.append({
                    "job_id": f"J{_job_id:05d}",
                    "customer_id": _cust["customer_id"],
                    "tech_id": _tech["tech_id"],
                    "service_type": _svc,
                    "job_type": _job_type,
                    "created_date": _created,
                    "scheduled_date": _date,
                    "completed_date": _date if _status == "Completed" else pd.NaT,
                    "duration_hours": round(_duration, 2),
                    "travel_time_hours": round(_travel, 2),
                    "parts_cost": round(_parts_cost, 2),
                    "labor_cost": round(_labor_cost, 2),
                    "total_revenue": round(_revenue, 2) if _status == "Completed" else 0.0,
                    "gross_profit": round(_gross_profit, 2),
                    "csat_rating": round(_csat, 1) if _status == "Completed" else None,
                    "first_time_fix": _ftf if _status == "Completed" else None,
                    "is_callback": _callback,
                    "on_time_arrival": _on_time if _status == "Completed" else None,
                    "status": _status,
                })

    jobs = pd.DataFrame(_jobs_list)
    completed_jobs = jobs[jobs["status"] == "Completed"].copy()

    return completed_jobs, customers, jobs, technicians


# =============================================================================
# MART CONSTRUCTION
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Analytical Marts

        The raw data is transformed into five analytical marts optimized for dashboarding
        and analysis. Each mart is a denormalized, aggregated table designed to answer a
        specific set of business questions without requiring complex joins at query time.

        | Mart | Grain | Purpose |
        |------|-------|---------|
        | `mart_daily_revenue` | Date x Service Type | Revenue trends, service mix, daily profitability |
        | `mart_monthly_kpis` | Month | Executive-level monthly KPI tracking |
        | `mart_technician_scorecard` | Month x Technician | Individual tech performance evaluation |
        | `mart_customer_health` | Customer | Lifetime value, retention, satisfaction analysis |
        | `mart_service_mix` | Month x Service Type | Service line performance comparison |

        ---
        """
    )
    return


@app.cell
def _(completed_jobs, customers, np, pd, technicians):
    # Working copy with time dimensions
    _cj = completed_jobs.copy()
    _cj["date"] = _cj["completed_date"].dt.date
    _cj["month_str"] = _cj["completed_date"].dt.strftime("%Y-%m")
    _cj["day_of_week"] = _cj["completed_date"].dt.day_name()
    _cj["week_num"] = _cj["completed_date"].dt.isocalendar().week.astype(int)

    # ---- mart_daily_revenue ----
    mart_daily_revenue = (
        _cj.groupby(["date", "service_type"])
        .agg(
            num_jobs=("job_id", "count"),
            total_revenue=("total_revenue", "sum"),
            total_parts_cost=("parts_cost", "sum"),
            total_labor_cost=("labor_cost", "sum"),
            gross_profit=("gross_profit", "sum"),
            avg_job_revenue=("total_revenue", "mean"),
            avg_csat=("csat_rating", "mean"),
            avg_duration=("duration_hours", "mean"),
        )
        .reset_index()
    )
    mart_daily_revenue["date"] = pd.to_datetime(mart_daily_revenue["date"])
    mart_daily_revenue["gross_margin_pct"] = (
        mart_daily_revenue["gross_profit"]
        / mart_daily_revenue["total_revenue"].replace(0, np.nan)
        * 100
    ).round(1)

    # ---- mart_monthly_kpis ----
    _monthly = _cj.groupby("month_str").agg(
        total_revenue=("total_revenue", "sum"),
        total_jobs=("job_id", "count"),
        avg_revenue_per_job=("total_revenue", "mean"),
        total_gross_profit=("gross_profit", "sum"),
        avg_csat=("csat_rating", "mean"),
        first_time_fix_rate=("first_time_fix", "mean"),
        on_time_rate=("on_time_arrival", "mean"),
        avg_duration=("duration_hours", "mean"),
        avg_travel_time=("travel_time_hours", "mean"),
        callback_count=("is_callback", "sum"),
        active_customers=("customer_id", "nunique"),
    ).reset_index()
    _monthly["gross_margin_pct"] = (
        _monthly["total_gross_profit"] / _monthly["total_revenue"] * 100
    ).round(1)
    _monthly["callback_rate"] = (
        _monthly["callback_count"] / _monthly["total_jobs"] * 100
    ).round(1)
    _monthly["first_time_fix_rate"] = (_monthly["first_time_fix_rate"] * 100).round(1)
    _monthly["on_time_rate"] = (_monthly["on_time_rate"] * 100).round(1)
    _monthly["avg_revenue_per_job"] = _monthly["avg_revenue_per_job"].round(2)
    _monthly["wrench_time_pct"] = (
        _monthly["avg_duration"]
        / (_monthly["avg_duration"] + _monthly["avg_travel_time"])
        * 100
    ).round(1)
    mart_monthly_kpis = _monthly

    # ---- mart_technician_scorecard ----
    _tech_m = _cj.groupby(["month_str", "tech_id"]).agg(
        jobs_completed=("job_id", "count"),
        revenue_generated=("total_revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_job_duration=("duration_hours", "mean"),
        avg_travel_time=("travel_time_hours", "mean"),
        first_time_fix_rate=("first_time_fix", "mean"),
        avg_csat=("csat_rating", "mean"),
        on_time_rate=("on_time_arrival", "mean"),
        callback_count=("is_callback", "sum"),
    ).reset_index()
    _tech_m = _tech_m.merge(
        technicians[["tech_id", "tech_name", "specialty", "skill_level", "hourly_rate"]],
        on="tech_id",
    )
    _tech_m["gross_margin_pct"] = (
        _tech_m["gross_profit"]
        / _tech_m["revenue_generated"].replace(0, np.nan)
        * 100
    ).round(1)
    _tech_m["first_time_fix_rate"] = (_tech_m["first_time_fix_rate"] * 100).round(1)
    _tech_m["on_time_rate"] = (_tech_m["on_time_rate"] * 100).round(1)
    # Utilization: billable hours / available hours (8h x ~22 work days)
    _tech_m["utilization_pct"] = (
        (_tech_m["avg_job_duration"] * _tech_m["jobs_completed"]) / (8 * 22) * 100
    ).round(1)
    mart_technician_scorecard = _tech_m

    # ---- mart_customer_health ----
    _cust_agg = _cj.groupby("customer_id").agg(
        total_jobs=("job_id", "count"),
        total_revenue=("total_revenue", "sum"),
        total_gross_profit=("gross_profit", "sum"),
        avg_csat=("csat_rating", "mean"),
        first_job_date=("completed_date", "min"),
        last_job_date=("completed_date", "max"),
        avg_revenue_per_job=("total_revenue", "mean"),
        first_time_fix_rate=("first_time_fix", "mean"),
        services_used=("service_type", "nunique"),
    ).reset_index()
    _cust_agg = _cust_agg.merge(
        customers[
            [
                "customer_id",
                "customer_name",
                "customer_type",
                "has_service_agreement",
                "agreement_monthly_value",
            ]
        ],
        on="customer_id",
    )
    _cust_agg["tenure_days"] = (
        _cust_agg["last_job_date"] - _cust_agg["first_job_date"]
    ).dt.days
    _cust_agg["is_repeat"] = _cust_agg["total_jobs"] >= 2
    _cust_agg["avg_csat"] = _cust_agg["avg_csat"].round(2)
    _cust_agg["first_time_fix_rate"] = (
        _cust_agg["first_time_fix_rate"] * 100
    ).round(1)
    mart_customer_health = _cust_agg

    # ---- mart_service_mix ----
    _svc_m = _cj.groupby(["month_str", "service_type"]).agg(
        num_jobs=("job_id", "count"),
        revenue=("total_revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_job_value=("total_revenue", "mean"),
        avg_duration=("duration_hours", "mean"),
        ftfr=("first_time_fix", "mean"),
        avg_csat=("csat_rating", "mean"),
    ).reset_index()
    _svc_total = _svc_m.groupby("month_str")["revenue"].transform("sum")
    _svc_m["pct_of_total_revenue"] = (_svc_m["revenue"] / _svc_total * 100).round(1)
    _svc_m["gross_margin_pct"] = (
        _svc_m["gross_profit"] / _svc_m["revenue"].replace(0, np.nan) * 100
    ).round(1)
    _svc_m["ftfr"] = (_svc_m["ftfr"] * 100).round(1)
    mart_service_mix = _svc_m

    # Collect all marts for persistence
    marts_dict = {
        "mart_daily_revenue": mart_daily_revenue,
        "mart_monthly_kpis": mart_monthly_kpis,
        "mart_technician_scorecard": mart_technician_scorecard,
        "mart_customer_health": mart_customer_health,
        "mart_service_mix": mart_service_mix,
    }

    return (
        mart_customer_health,
        mart_daily_revenue,
        mart_monthly_kpis,
        mart_service_mix,
        mart_technician_scorecard,
        marts_dict,
    )


# =============================================================================
# DLT PERSISTENCE TO DUCKDB
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Persistent Storage via dlt

        All analytical marts are saved to a local DuckDB database using
        [dlt (data load tool)](https://dlthub.com/). This creates a queryable
        analytical warehouse that persists across notebook sessions and can be
        connected to external BI tools (Metabase, Superset, DBeaver, etc.).

        ---
        """
    )
    return


@app.cell
def _(dlt, marts_dict, mo, pd):
    _pipeline = dlt.pipeline(
        pipeline_name="field_services",
        destination="duckdb",
        dataset_name="marts",
    )

    _load_results = []
    for _name, _df in marts_dict.items():
        # Clean DataFrame for dlt compatibility
        _clean = _df.copy()
        for _col in _clean.columns:
            # Convert date objects to datetime for dlt
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
        _load_results.append(f"| `{_name}` | {len(_df):,} rows | Loaded |")

    _db_path = str(_pipeline.destination_client().config)

    dlt_load_info = mo.md(
        f"""
        **Database:** `{_db_path}`

        | Mart | Size | Status |
        |------|------|--------|
        {"".join(chr(10) + r for r in _load_results)}

        All marts written with `write_disposition='replace'` -- re-running this cell
        refreshes the database with the latest generated data.
        """
    )
    dlt_load_info
    return (dlt_load_info,)


# =============================================================================
# INTERACTIVE DASHBOARDS
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Interactive Dashboards

        Five dashboards organized by stakeholder need. Use the filters below to
        focus on specific service types or technicians.

        ---
        """
    )
    return


@app.cell
def _(SERVICE_TYPES, mo, technicians):
    service_filter = mo.ui.multiselect(
        options=SERVICE_TYPES,
        value=SERVICE_TYPES,
        label="Service Types",
    )
    tech_selector = mo.ui.dropdown(
        options=dict(
            zip(
                technicians["tech_name"].tolist(),
                technicians["tech_id"].tolist(),
            )
        ),
        value=technicians["tech_name"].iloc[0],
        label="Technician",
    )
    mo.hstack(
        [service_filter, tech_selector],
        justify="start",
        gap=1.5,
    )
    return service_filter, tech_selector


# =============================================================================
# EXECUTIVE SUMMARY DASHBOARD
# =============================================================================


@app.cell
def _(
    COLORS,
    SERVICE_COLORS,
    apply_chart_style,
    go,
    mart_monthly_kpis,
    mart_service_mix,
    mo,
    service_filter,
):
    # Filter service mix by selected services
    _mix = mart_service_mix[
        mart_service_mix["service_type"].isin(service_filter.value)
    ]

    # Latest and prior month for KPI deltas
    _latest = mart_monthly_kpis.iloc[-1]
    _prev = mart_monthly_kpis.iloc[-2] if len(mart_monthly_kpis) > 1 else _latest

    # ---- KPI Cards ----
    def _kpi(title, val, prev_val, prefix="", suffix="", fmt=",.0f"):
        _f = go.Figure(
            go.Indicator(
                mode="number+delta",
                value=val,
                number={"prefix": prefix, "suffix": suffix, "valueformat": fmt, "font": {"size": 30}},
                delta={"reference": prev_val, "relative": True, "valueformat": ".1%"},
                title={"text": title, "font": {"size": 13, "color": "#555"}},
            )
        )
        _f.update_layout(
            height=120,
            margin=dict(t=45, b=10, l=15, r=15),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return _f

    _k1 = _kpi("Monthly Revenue", _latest["total_revenue"], _prev["total_revenue"], prefix="$")
    _k2 = _kpi("Completed Jobs", _latest["total_jobs"], _prev["total_jobs"])
    _k3 = _kpi("Avg Revenue/Job", _latest["avg_revenue_per_job"], _prev["avg_revenue_per_job"], prefix="$")
    _k4 = _kpi("Gross Margin", _latest["gross_margin_pct"], _prev["gross_margin_pct"], suffix="%", fmt=".1f")
    _k5 = _kpi("First-Time Fix", _latest["first_time_fix_rate"], _prev["first_time_fix_rate"], suffix="%", fmt=".1f")
    _k6 = _kpi("Avg CSAT", _latest["avg_csat"], _prev["avg_csat"], fmt=".2f")

    # ---- Revenue Trend ----
    _rev = go.Figure()
    _rev.add_trace(
        go.Scatter(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["total_revenue"],
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=7),
            name="Revenue",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )
    _rev.add_trace(
        go.Scatter(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["total_gross_profit"],
            mode="lines+markers",
            line=dict(color=COLORS["success"], width=2, dash="dash"),
            marker=dict(size=5),
            name="Gross Profit",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )
    apply_chart_style(_rev, "Monthly Revenue & Gross Profit", height=350)
    _rev.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Service Mix Donut ----
    _latest_month = mart_monthly_kpis["month_str"].iloc[-1]
    _lm = _mix[_mix["month_str"] == _latest_month]
    _donut = go.Figure(
        go.Pie(
            labels=_lm["service_type"],
            values=_lm["revenue"],
            hole=0.5,
            marker=dict(
                colors=[SERVICE_COLORS.get(s, "#999") for s in _lm["service_type"]]
            ),
            textinfo="label+percent",
            hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
        )
    )
    apply_chart_style(_donut, f"Revenue Mix ({_latest_month})", height=350)

    # ---- Jobs & Margin Dual Axis ----
    _jm = go.Figure()
    _jm.add_trace(
        go.Bar(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["total_jobs"],
            name="Jobs Completed",
            marker_color=COLORS["secondary"],
            opacity=0.8,
            yaxis="y",
            hovertemplate="%{y:,} jobs<extra></extra>",
        )
    )
    _jm.add_trace(
        go.Scatter(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["gross_margin_pct"],
            name="Gross Margin %",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=7),
            yaxis="y2",
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    apply_chart_style(_jm, "Jobs Completed & Gross Margin Trend", height=350)
    _jm.update_layout(
        yaxis=dict(title="Jobs", side="left"),
        yaxis2=dict(title="Margin %", side="right", overlaying="y", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    executive_content = mo.vstack([
        mo.hstack([_k1, _k2, _k3, _k4, _k5, _k6], justify="space-around", widths="equal"),
        mo.hstack([_rev, _donut], widths="equal"),
        _jm,
    ])
    return (executive_content,)


# =============================================================================
# REVENUE & PROFITABILITY DASHBOARD
# =============================================================================


@app.cell
def _(
    SERVICE_COLORS,
    apply_chart_style,
    go,
    mart_customer_health,
    mart_service_mix,
    mo,
    px,
    service_filter,
):
    _mix = mart_service_mix[
        mart_service_mix["service_type"].isin(service_filter.value)
    ]

    # ---- Revenue by Service Type (stacked area) ----
    _area = go.Figure()
    for _svc in service_filter.value:
        _svc_data = _mix[_mix["service_type"] == _svc].sort_values("month_str")
        _area.add_trace(
            go.Scatter(
                x=_svc_data["month_str"],
                y=_svc_data["revenue"],
                mode="lines",
                stackgroup="one",
                name=_svc,
                line=dict(color=SERVICE_COLORS.get(_svc, "#999")),
                hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
            )
        )
    apply_chart_style(_area, "Revenue by Service Type", height=380)
    _area.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Profitability by Service Type ----
    _latest_month = _mix["month_str"].max()
    _prof = _mix[_mix["month_str"] == _latest_month].copy()
    _prof_fig = go.Figure()
    _prof_fig.add_trace(
        go.Bar(
            x=_prof["service_type"],
            y=_prof["avg_job_value"],
            name="Avg Job Value",
            marker_color=[SERVICE_COLORS.get(s, "#999") for s in _prof["service_type"]],
            hovertemplate="$%{y:,.0f}<extra>Avg Job Value</extra>",
        )
    )
    _prof_fig.add_trace(
        go.Scatter(
            x=_prof["service_type"],
            y=_prof["gross_margin_pct"],
            name="Gross Margin %",
            mode="markers+text",
            marker=dict(size=14, color="#E74C3C", symbol="diamond"),
            text=[f"{v:.0f}%" for v in _prof["gross_margin_pct"]],
            textposition="top center",
            yaxis="y2",
        )
    )
    apply_chart_style(_prof_fig, f"Job Value vs. Margin by Service ({_latest_month})", height=380)
    _prof_fig.update_layout(
        yaxis=dict(title="Avg Job Value ($)", tickprefix="$"),
        yaxis2=dict(title="Margin %", side="right", overlaying="y", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Top 15 Customers by Revenue ----
    _top_cust = (
        mart_customer_health
        .nlargest(15, "total_revenue")
        .sort_values("total_revenue")
    )
    _top_fig = px.bar(
        _top_cust,
        x="total_revenue",
        y="customer_name",
        color="customer_type",
        orientation="h",
        color_discrete_map={"Residential": "#2E86AB", "Commercial": "#F57C20"},
        hover_data={"total_jobs": True, "avg_csat": ":.1f"},
    )
    apply_chart_style(_top_fig, "Top 15 Customers by Revenue", height=420)
    _top_fig.update_layout(
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Monthly Revenue Growth ----
    _growth = _mix.groupby("month_str")["revenue"].sum().reset_index()
    _growth["prev"] = _growth["revenue"].shift(1)
    _growth["growth_pct"] = (
        (_growth["revenue"] - _growth["prev"]) / _growth["prev"] * 100
    ).round(1)
    _growth = _growth.dropna()
    _grow_fig = go.Figure(
        go.Bar(
            x=_growth["month_str"],
            y=_growth["growth_pct"],
            marker_color=[
                "#27AE60" if v >= 0 else "#E74C3C" for v in _growth["growth_pct"]
            ],
            hovertemplate="%{y:+.1f}%<extra></extra>",
        )
    )
    apply_chart_style(_grow_fig, "Month-over-Month Revenue Growth %", height=350)
    _grow_fig.update_layout(yaxis_ticksuffix="%")

    revenue_content = mo.vstack([
        mo.hstack([_area, _prof_fig], widths="equal"),
        mo.hstack([_top_fig, _grow_fig], widths="equal"),
    ])
    return (revenue_content,)


# =============================================================================
# TECHNICIAN PERFORMANCE DASHBOARD
# =============================================================================


@app.cell
def _(
    COLORS,
    apply_chart_style,
    go,
    mart_technician_scorecard,
    mo,
    px,
    tech_selector,
):
    _ts = mart_technician_scorecard.copy()
    _latest_month = _ts["month_str"].max()

    # ---- KPIs for selected technician ----
    _sel_id = tech_selector.value
    _sel = _ts[(_ts["tech_id"] == _sel_id) & (_ts["month_str"] == _latest_month)]

    if len(_sel) > 0:
        _s = _sel.iloc[0]
        _sel_name = _s["tech_name"]

        def _tkpi(title, val, fmt=",.0f", prefix="", suffix=""):
            _f = go.Figure(
                go.Indicator(
                    mode="number",
                    value=val,
                    number={"prefix": prefix, "suffix": suffix, "valueformat": fmt, "font": {"size": 28}},
                    title={"text": title, "font": {"size": 12, "color": "#555"}},
                )
            )
            _f.update_layout(height=110, margin=dict(t=40, b=5, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
            return _f

        _tk1 = _tkpi("Jobs", _s["jobs_completed"])
        _tk2 = _tkpi("Revenue", _s["revenue_generated"], prefix="$")
        _tk3 = _tkpi("FTFR", _s["first_time_fix_rate"], suffix="%", fmt=".1f")
        _tk4 = _tkpi("CSAT", _s["avg_csat"], fmt=".2f")
        _tk5 = _tkpi("Margin", _s["gross_margin_pct"], suffix="%", fmt=".1f")

        _tech_kpis = mo.vstack([
            mo.md(f"**{_sel_name}** -- {_s['specialty']} (Skill Level {_s['skill_level']}/5) -- {_latest_month}"),
            mo.hstack([_tk1, _tk2, _tk3, _tk4, _tk5], justify="space-around", widths="equal"),
        ])
    else:
        _tech_kpis = mo.md("*No data for selected technician this month.*")

    # ---- All Techs: Revenue Comparison (latest month) ----
    _all_latest = _ts[_ts["month_str"] == _latest_month].sort_values(
        "revenue_generated", ascending=True
    )
    _rev_bar = px.bar(
        _all_latest,
        x="revenue_generated",
        y="tech_name",
        orientation="h",
        color="specialty",
        hover_data={"jobs_completed": True, "gross_margin_pct": ":.1f"},
    )
    apply_chart_style(_rev_bar, f"Revenue by Technician ({_latest_month})", height=420)
    _rev_bar.update_layout(
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- FTFR by Technician ----
    _ftfr_bar = go.Figure(
        go.Bar(
            x=_all_latest["tech_name"],
            y=_all_latest["first_time_fix_rate"],
            marker_color=[
                COLORS["success"] if v >= 80 else COLORS["warning"] if v >= 70 else COLORS["danger"]
                for v in _all_latest["first_time_fix_rate"]
            ],
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    apply_chart_style(_ftfr_bar, f"First-Time Fix Rate by Technician ({_latest_month})", height=380)
    _ftfr_bar.update_layout(yaxis_ticksuffix="%", yaxis_range=[0, 100])
    _ftfr_bar.add_hline(y=80, line_dash="dash", line_color="#999", annotation_text="80% Target")

    # ---- CSAT by Technician ----
    _csat_bar = go.Figure(
        go.Bar(
            x=_all_latest["tech_name"],
            y=_all_latest["avg_csat"],
            marker_color=[
                COLORS["success"] if v >= 4.0 else COLORS["warning"] if v >= 3.5 else COLORS["danger"]
                for v in _all_latest["avg_csat"]
            ],
            hovertemplate="%{y:.2f}<extra></extra>",
        )
    )
    apply_chart_style(_csat_bar, f"Average CSAT by Technician ({_latest_month})", height=380)
    _csat_bar.update_layout(yaxis_range=[1, 5])
    _csat_bar.add_hline(y=4.0, line_dash="dash", line_color="#999", annotation_text="4.0 Target")

    tech_content = mo.vstack([
        _tech_kpis,
        _rev_bar,
        mo.hstack([_ftfr_bar, _csat_bar], widths="equal"),
    ])
    return (tech_content,)


# =============================================================================
# CUSTOMER HEALTH DASHBOARD
# =============================================================================


@app.cell
def _(
    COLORS,
    apply_chart_style,
    go,
    mart_customer_health,
    mo,
    px,
):
    _ch = mart_customer_health.copy()

    # ---- Customer Segment Distribution ----
    _seg = _ch.groupby("customer_type").agg(
        count=("customer_id", "count"),
        total_revenue=("total_revenue", "sum"),
        avg_csat=("avg_csat", "mean"),
    ).reset_index()

    _seg_fig = go.Figure()
    _seg_fig.add_trace(
        go.Bar(
            x=_seg["customer_type"],
            y=_seg["count"],
            name="Customer Count",
            marker_color=COLORS["primary"],
            yaxis="y",
        )
    )
    _seg_fig.add_trace(
        go.Bar(
            x=_seg["customer_type"],
            y=_seg["total_revenue"],
            name="Total Revenue",
            marker_color=COLORS["accent"],
            yaxis="y2",
        )
    )
    apply_chart_style(_seg_fig, "Customer Segments: Count vs. Revenue", height=380)
    _seg_fig.update_layout(
        yaxis=dict(title="Customers", side="left"),
        yaxis2=dict(title="Revenue ($)", side="right", overlaying="y", tickprefix="$", tickformat=","),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Lifetime Value Distribution ----
    _clv_fig = px.histogram(
        _ch,
        x="total_revenue",
        color="customer_type",
        nbins=40,
        color_discrete_map={"Residential": "#2E86AB", "Commercial": "#F57C20"},
        marginal="box",
    )
    apply_chart_style(_clv_fig, "Customer Lifetime Revenue Distribution", height=380)
    _clv_fig.update_layout(
        xaxis_title="Total Revenue ($)",
        xaxis_tickprefix="$",
        yaxis_title="Customer Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Agreement Impact ----
    _agr = _ch.groupby("has_service_agreement").agg(
        avg_revenue=("total_revenue", "mean"),
        avg_jobs=("total_jobs", "mean"),
        avg_csat=("avg_csat", "mean"),
        retention=("is_repeat", "mean"),
    ).reset_index()
    _agr["has_service_agreement"] = _agr["has_service_agreement"].map(
        {True: "With Agreement", False: "No Agreement"}
    )
    _agr["retention"] = (_agr["retention"] * 100).round(1)

    _agr_fig = go.Figure()
    _metrics = ["avg_revenue", "avg_jobs", "avg_csat", "retention"]
    _labels = ["Avg Revenue ($)", "Avg Jobs", "Avg CSAT", "Repeat Rate (%)"]
    for _i, (_m, _l) in enumerate(zip(_metrics, _labels)):
        _agr_fig.add_trace(
            go.Bar(
                x=_agr["has_service_agreement"],
                y=_agr[_m],
                name=_l,
                text=[f"{v:.1f}" for v in _agr[_m]],
                textposition="outside",
            )
        )
    apply_chart_style(_agr_fig, "Service Agreement Impact on Key Metrics", height=380)
    _agr_fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Top Customers Table ----
    _top = _ch.nlargest(20, "total_revenue")[
        [
            "customer_name",
            "customer_type",
            "total_jobs",
            "total_revenue",
            "avg_csat",
            "has_service_agreement",
            "is_repeat",
        ]
    ].copy()
    _top["total_revenue"] = _top["total_revenue"].apply(lambda x: f"${x:,.0f}")
    _top.columns = [
        "Customer",
        "Type",
        "Jobs",
        "Revenue",
        "CSAT",
        "Agreement",
        "Repeat",
    ]
    _table = mo.ui.table(_top, label="Top 20 Customers by Revenue")

    customer_content = mo.vstack([
        mo.hstack([_seg_fig, _clv_fig], widths="equal"),
        _agr_fig,
        _table,
    ])
    return (customer_content,)


# =============================================================================
# OPERATIONS DASHBOARD
# =============================================================================


@app.cell
def _(
    COLORS,
    SERVICE_COLORS,
    apply_chart_style,
    completed_jobs,
    go,
    mart_monthly_kpis,
    mo,
    px,
    service_filter,
):
    _ops = completed_jobs[
        completed_jobs["service_type"].isin(service_filter.value)
    ].copy()
    _ops["day_of_week"] = _ops["completed_date"].dt.day_name()
    _ops["hour"] = (_ops["travel_time_hours"] * 3 + 8).astype(int).clip(7, 18)
    _ops["month_str"] = _ops["completed_date"].dt.strftime("%Y-%m")

    # ---- On-Time & FTFR Trends ----
    _trend = go.Figure()
    _trend.add_trace(
        go.Scatter(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["on_time_rate"],
            mode="lines+markers",
            name="On-Time Rate",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=7),
        )
    )
    _trend.add_trace(
        go.Scatter(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["first_time_fix_rate"],
            mode="lines+markers",
            name="First-Time Fix Rate",
            line=dict(color=COLORS["success"], width=3),
            marker=dict(size=7),
        )
    )
    _trend.add_trace(
        go.Scatter(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["callback_rate"],
            mode="lines+markers",
            name="Callback Rate",
            line=dict(color=COLORS["danger"], width=2, dash="dash"),
            marker=dict(size=5),
        )
    )
    apply_chart_style(_trend, "Operational Quality Trends", height=380)
    _trend.update_layout(
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Wrench Time vs Travel Time ----
    _wt = go.Figure()
    _wt.add_trace(
        go.Bar(
            x=mart_monthly_kpis["month_str"],
            y=mart_monthly_kpis["wrench_time_pct"],
            name="Wrench Time %",
            marker_color=COLORS["success"],
        )
    )
    _wt.add_trace(
        go.Bar(
            x=mart_monthly_kpis["month_str"],
            y=100 - mart_monthly_kpis["wrench_time_pct"],
            name="Travel/Other %",
            marker_color=COLORS["muted"],
        )
    )
    apply_chart_style(_wt, "Wrench Time vs. Travel/Other", height=380)
    _wt.update_layout(
        barmode="stack",
        yaxis_ticksuffix="%",
        yaxis_range=[0, 100],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Jobs by Day of Week ----
    _dow_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    _dow = (
        _ops.groupby("day_of_week")
        .agg(
            num_jobs=("job_id", "count"),
            avg_revenue=("total_revenue", "mean"),
        )
        .reindex(_dow_order)
        .reset_index()
    )
    _dow_fig = go.Figure()
    _dow_fig.add_trace(
        go.Bar(
            x=_dow["day_of_week"],
            y=_dow["num_jobs"],
            name="Total Jobs",
            marker_color=COLORS["secondary"],
        )
    )
    apply_chart_style(_dow_fig, "Job Volume by Day of Week", height=380)

    # ---- Duration by Service Type (box plot) ----
    _dur_fig = px.box(
        _ops,
        x="service_type",
        y="duration_hours",
        color="service_type",
        color_discrete_map=SERVICE_COLORS,
    )
    apply_chart_style(_dur_fig, "Job Duration Distribution by Service Type", height=380)
    _dur_fig.update_layout(
        xaxis_title="",
        yaxis_title="Duration (hours)",
        showlegend=False,
    )

    ops_content = mo.vstack([
        mo.hstack([_trend, _wt], widths="equal"),
        mo.hstack([_dow_fig, _dur_fig], widths="equal"),
    ])
    return (ops_content,)


# =============================================================================
# DASHBOARD TABS
# =============================================================================


@app.cell
def _(
    customer_content,
    executive_content,
    mo,
    ops_content,
    revenue_content,
    tech_content,
):
    _tabs = mo.ui.tabs({
        "Executive Summary": executive_content,
        "Revenue & Profitability": revenue_content,
        "Technician Performance": tech_content,
        "Customer Intelligence": customer_content,
        "Operations": ops_content,
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
        ## 7. Data Explorer

        Use the interactive table below to explore the raw analytical marts directly.
        Click column headers to sort, and use the search bar to filter.

        ---
        """
    )
    return


@app.cell
def _(marts_dict, mo):
    mart_selector = mo.ui.dropdown(
        options=list(marts_dict.keys()),
        value="mart_monthly_kpis",
        label="Select Mart",
    )
    mart_selector
    return (mart_selector,)


@app.cell
def _(mart_selector, marts_dict, mo):
    _selected_df = marts_dict[mart_selector.value]
    mo.vstack([
        mo.md(
            f"**{mart_selector.value}** -- {len(_selected_df):,} rows, "
            f"{len(_selected_df.columns)} columns"
        ),
        mo.ui.table(_selected_df, page_size=20, label=mart_selector.value),
    ])
    return


# =============================================================================
# FOOTER
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        **Field Services Analytics Platform** | Built with
        [marimo](https://marimo.io), [Plotly](https://plotly.com/python/),
        and [dlt](https://dlthub.com/)

        > **Next steps for production deployment:**
        >
        > - Replace synthetic data with live API connections (ServiceTitan, Housecall Pro,
        >   Jobber, or your FSM platform)
        > - Schedule daily mart refreshes via `dlt` pipelines
        > - Add alerting thresholds for FTFR < 75%, CSAT < 4.0, and margin < 40%
        > - Extend with technician GPS/routing data for travel time optimization
        > - Add financial forecasting using historical seasonal patterns
        """
    )
    return


if __name__ == "__main__":
    app.run()
