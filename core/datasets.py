"""
Core Causal Inference & Experimentation Platform Engine
Module: Realistic Google-Scale Synthetic Datasets
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def generate_search_ranking_experiment(n_users: int = 40000, seed: int = 42) -> pd.DataFrame:
    """
    Simulates a Google Search Ranking latency & engagement A/B test.
    High pre/post correlation (rho ~ 0.78) to showcase CUPED sample size compression.
    """
    rng = np.random.default_rng(seed)

    user_ids = [f"user_{i:07d}" for i in range(1, n_users + 1)]
    # 50/50 randomized split
    variants = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    is_treatment = (variants == "treatment").astype(float)

    # Pre-experiment activity (14-day baseline search queries)
    # Log-normal distribution typical of search volume
    baseline_activity = rng.lognormal(mean=2.8, sigma=0.75, size=n_users)
    pre_queries = np.round(np.clip(baseline_activity, 1, 150), 1)

    # Post-experiment activity: high correlation with pre-period + true treatment lift of +4.5%
    noise = rng.normal(loc=0, scale=3.5, size=n_users)
    true_lift = 0.045 * pre_queries * is_treatment
    post_queries = np.round(np.clip(0.85 * pre_queries + 3.0 + true_lift + noise, 1, 180), 1)

    # Secondary metric: Search latency in milliseconds
    # Treatment introduced optimized model reducing latency by 28ms
    base_latency = rng.normal(loc=195, scale=30, size=n_users)
    latency_ms = np.round(np.clip(base_latency - 28.0 * is_treatment, 50, 400), 1)

    # User satisfaction score (CSAT 1.0 to 5.0)
    csat = np.round(np.clip(4.1 + 0.15 * is_treatment + rng.normal(0, 0.4, n_users), 1.0, 5.0), 2)

    df = pd.DataFrame({
        "user_id": user_ids,
        "variant": variants,
        "is_treatment": is_treatment.astype(int),
        "pre_experiment_queries": pre_queries,
        "post_experiment_queries": post_queries,
        "latency_ms": latency_ms,
        "satisfaction_score": csat,
    })
    return df


def generate_srm_bug_dataset(n_users: int = 50000, seed: int = 101) -> pd.DataFrame:
    """
    Simulates a production A/B experiment contaminated by an insidious Sample Ratio Mismatch (SRM).
    Overall split appears biased (~48.2% vs 51.8%, p < 1e-12).
    Slice analysis isolates the bug to Mobile Safari users where a faulty redirect script caused telemetry loss.
    """
    rng = np.random.default_rng(seed)

    user_ids = [f"usr_{i:07d}" for i in range(1, n_users + 1)]
    devices = rng.choice(["Mobile", "Desktop", "Tablet"], size=n_users, p=[0.60, 0.32, 0.08])
    browsers = rng.choice(["Chrome", "Safari", "Firefox", "Edge"], size=n_users, p=[0.55, 0.30, 0.09, 0.06])
    countries = rng.choice(["US", "UK", "DE", "JP", "IN"], size=n_users, p=[0.40, 0.20, 0.15, 0.15, 0.10])
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-10-01", periods=14, freq="D")]
    assigned_dates = rng.choice(dates, size=n_users)

    # Initial designed 50/50 assignment
    initial_treatment = rng.binomial(1, 0.5, size=n_users)

    # Faulty client-side bug: Mobile Safari treatment users experienced redirect failure (drop rate ~35%)
    keep_mask = np.ones(n_users, dtype=bool)
    for i in range(n_users):
        if devices[i] == "Mobile" and browsers[i] == "Safari" and initial_treatment[i] == 1:
            if rng.random() < 0.38:  # 38% drop rate on Mobile Safari Treatment
                keep_mask[i] = False

    df = pd.DataFrame({
        "user_id": np.array(user_ids)[keep_mask],
        "date": assigned_dates[keep_mask],
        "variant": ["treatment" if t == 1 else "control" for t in initial_treatment[keep_mask]],
        "device_type": devices[keep_mask],
        "browser": browsers[keep_mask],
        "country": countries[keep_mask],
    })
    return df


def generate_regional_rollout_data(
    n_regions: int = 12,
    n_weeks: int = 52,
    treat_week: int = 36,
    treated_region: str = "US-West",
    seed: int = 777,
) -> pd.DataFrame:
    """
    Simulates weekly panel data for an aggregate regional rollout (e.g. Android AI feature launch).
    Contains common seasonal trends, region-specific baseline fixed effects, and parallel pre-trends.
    Post-week 36, the treated region experiences a sharp +8.2% lift.
    """
    rng = np.random.default_rng(seed)

    regions = [
        "US-West", "US-East", "US-Central", "US-South",
        "CA-Ontario", "UK-South", "DE-Bavaria", "FR-IleDeFrance",
        "JP-Kanto", "AU-NSW", "BR-SaoPaulo", "SG-Central"
    ]
    if treated_region not in regions:
        regions[0] = treated_region

    weeks = list(range(1, n_weeks + 1))
    records = []

    # Common macro seasonality (holiday peaks, weekly cycles)
    macro_trend = 100.0 + 15.0 * np.sin(np.linspace(0, 3 * np.pi, n_weeks)) + 0.3 * np.array(weeks)

    # Region-specific baseline scales
    region_baselines = {
        r: rng.uniform(85.0, 140.0) for r in regions
    }
    # Treated region baseline
    region_baselines[treated_region] = 125.0

    for r in regions:
        is_treated_unit = int(r == treated_region)
        base = region_baselines[r]

        for w in weeks:
            is_post = int(w >= treat_week)
            noise = rng.normal(0, 1.8)

            # Causal lift on treated region starting at treat_week
            if is_treated_unit and is_post:
                # Gradual ramp-up: reaches full lift after 3 weeks
                ramp = min(1.0, (w - treat_week + 1) / 3.0)
                lift = ramp * 14.5
            else:
                lift = 0.0

            metric = (base / 100.0) * macro_trend[w - 1] + lift + noise
            # Supporting covariate (e.g., active user volume index)
            covariate = (base / 100.0) * (macro_trend[w - 1] * 0.7) + rng.normal(0, 2.0)

            records.append({
                "region": r,
                "week": w,
                "date": f"2026-W{w:02d}",
                "is_treated_region": is_treated_unit,
                "is_post_period": is_post,
                "engagement_score": round(float(metric), 2),
                "active_user_index": round(float(covariate), 2),
            })

    return pd.DataFrame(records)


def generate_youtube_causal_data(n_users: int = 15000, seed: int = 42) -> pd.DataFrame:
    """
    Simulates observational data for YouTube recommendation algorithm evaluation.
    Treatment is non-randomly assigned based on confounders (tenure, network speed).
    """
    rng = np.random.default_rng(seed)

    user_ids = [f"yt_user_{i:06d}" for i in range(1, n_users + 1)]

    # Confounder 1: Account tenure in months (older accounts use higher bandwidth)
    tenure_months = rng.gamma(shape=3.0, scale=10.0, size=n_users)

    # Confounder 2: Network connection speed in Mbps
    network_speed_mbps = np.clip(rng.normal(loc=45, scale=20, size=n_users), 5, 150)

    # Confounder 3: Device screen size category
    screen_category = rng.choice([1, 2, 3], size=n_users, p=[0.25, 0.45, 0.30])  # Mobile, Tablet, TV

    # Treatment assignment: modern neural recommendation model (v2) vs baseline (v1)
    # Treatment probability is higher for users with longer tenure and faster connection
    logit_p = -1.2 + 0.03 * tenure_months + 0.02 * network_speed_mbps + 0.25 * screen_category
    propensity = 1.0 / (1.0 + np.exp(-logit_p))
    treatment = rng.binomial(1, propensity)

    # Outcome: Weekly watch time in hours
    # True causal effect of treatment is +3.5 hours
    # Confounders also directly increase watch time (older accounts and faster speed watch more)
    base_watch_time = 8.0 + 0.15 * tenure_months + 0.08 * network_speed_mbps + 2.5 * screen_category
    true_causal_effect = 3.5 * treatment
    noise = rng.normal(0, 4.0, size=n_users)

    watch_time_hours = np.round(np.clip(base_watch_time + true_causal_effect + noise, 0.5, 80.0), 1)

    df = pd.DataFrame({
        "user_id": user_ids,
        "recommendation_v2": treatment,
        "watch_time_hours": watch_time_hours,
        "tenure_months": np.round(tenure_months, 1),
        "network_speed_mbps": np.round(network_speed_mbps, 1),
        "screen_category": screen_category,
    })
    return df
