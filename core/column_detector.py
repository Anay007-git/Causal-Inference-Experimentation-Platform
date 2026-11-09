"""
Core Causal Inference & Experimentation Platform Engine
Module: Intelligent Schema Auto-Detection & Data Mapping Wizard
Enables the platform to automatically detect experiment structure and ingest arbitrary user datasets.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
import numpy as np
import pandas as pd


def detect_variant_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intelligently identifies the experimental variant or treatment assignment column.
    """
    if df.empty:
        return None
    cols = list(df.columns)

    # 1. High-confidence regex match on column names
    patterns = [
        r"^(is_)?treat(ment)?(_group|_flag|_ind)?$",
        r"^(test_)?variant$",
        r"^group$",
        r"^arm$",
        r"^(test_)?condition$",
        r"^cohort$",
        r"^split$",
        r"^bucket$",
        r"^is_treated(_region)?$",
    ]
    for p in patterns:
        for c in cols:
            if re.search(p, str(c).strip().lower()):
                if 2 <= df[c].nunique(dropna=True) <= 10:
                    return c

    # 2. Semantic value check: columns with values like ('control', 'treatment'), ('A', 'B'), etc.
    canonical_pairs = [
        {"control", "treatment"},
        {"ctrl", "treat"},
        {"a", "b"},
        {"test", "control"},
        {"baseline", "variant"},
    ]
    for c in cols:
        vals = set(str(v).strip().lower() for v in df[c].dropna().unique())
        for pair in canonical_pairs:
            if pair.issubset(vals) or vals == pair:
                return c

    # 3. Any boolean / 0-1 binary column
    for c in cols:
        nunique = df[c].nunique(dropna=True)
        if nunique == 2:
            vals = set(df[c].dropna().unique())
            if vals in [{0, 1}, {False, True}, {"0", "1"}, {"True", "False"}]:
                return c

    # 4. Fallback: Any low-cardinality categorical column (2 to 4 unique values)
    low_card = [c for c in cols if 2 <= df[c].nunique(dropna=True) <= 4 and not np.issubdtype(df[c].dtype, np.number)]
    if low_card:
        return low_card[0]

    return None


def detect_outcome_column(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Optional[str]:
    """
    Detects the primary continuous or count metric outcome column (Y).
    """
    if df.empty:
        return None
    exclude_cols = exclude_cols or []
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in exclude_cols]
    if not numeric_cols:
        return None

    # 1. Regex match for common outcome metric naming conventions
    patterns = [
        r"^post_.*(metric|queries|clicks|revenue|spend|score|engagement|val)",
        r".*(revenue|spend|conversion|watch_time|queries|clicks|score|orders|purchase|gmv|duration|dwell)",
        r"^(primary_)?(outcome|metric|y|target|result|kpi)$",
    ]
    for p in patterns:
        for c in numeric_cols:
            if re.search(p, str(c).strip().lower()):
                return c

    # 2. Pick the numeric column with the greatest standard deviation or variance
    stds = df[numeric_cols].std(numeric_only=True)
    if not stds.empty and stds.max() > 0:
        return stds.idxmax()

    return numeric_cols[-1]


def detect_baseline_covariate_columns(
    df: pd.DataFrame,
    outcome_col: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Detects pre-experiment baseline features / covariates (X) for CUPED variance reduction.
    """
    if df.empty:
        return []
    exclude_cols = list(exclude_cols or [])
    if outcome_col and outcome_col not in exclude_cols:
        exclude_cols.append(outcome_col)

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in exclude_cols]
    if not numeric_cols:
        return []

    # 1. Regex patterns for pre-experiment signals
    pre_cols = []
    patterns = [r"^pre_.*", r"^baseline_.*", r"^prior_.*", r"^historical_.*", r"^past_.*", r"^lag_.*"]
    for p in patterns:
        for c in numeric_cols:
            if re.search(p, str(c).strip().lower()) and c not in pre_cols:
                pre_cols.append(c)

    if pre_cols:
        return pre_cols

    # 2. Correlation ranking with outcome_col
    if outcome_col and outcome_col in df.columns and np.issubdtype(df[outcome_col].dtype, np.number):
        corrs = {}
        for c in numeric_cols:
            try:
                corr = df[[outcome_col, c]].corr().iloc[0, 1]
                if not np.isnan(corr) and abs(corr) > 0.05:
                    corrs[c] = abs(corr)
            except Exception:
                pass
        if corrs:
            # Sort by correlation descending
            sorted_by_corr = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
            return [k for k, _ in sorted_by_corr[:3]]

    return numeric_cols[:1] if numeric_cols else []


def detect_unit_column(df: pd.DataFrame, prefer_regional: bool = True) -> Optional[str]:
    """
    Detects unit / region / entity identifier column.
    If prefer_regional is True, prioritizes macro-level entities (regions, markets, states).
    """
    if df.empty:
        return None
    cols = list(df.columns)

    if prefer_regional:
        # Regional / cohort names
        regional_patterns = [
            r"^(geo_)?region$",
            r"^market$",
            r"^state$",
            r"^country(_code)?$",
            r"^city$",
            r"^store(_id)?$",
            r"^cluster$",
            r"^unit$",
        ]
        for p in regional_patterns:
            for c in cols:
                if re.search(p, str(c).strip().lower()):
                    if 2 <= df[c].nunique(dropna=True) <= 150:
                        return c

    # Generic entity ID
    id_patterns = [
        r"^(user_)?id$",
        r"^uid$",
        r"^account_id$",
        r"^customer_id$",
        r"^device_id$",
        r"^visitor_id$",
    ]
    for p in id_patterns:
        for c in cols:
            if re.search(p, str(c).strip().lower()):
                return c

    # Fallback to first column
    return cols[0] if cols else None


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detects timestamp, date, or relative period column.
    """
    if df.empty:
        return None
    cols = list(df.columns)

    # 1. Date/time dtypes
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c

    # 2. Regex matching time-related names
    time_patterns = [
        r"^week(_id)?$",
        r"^(event_)?date$",
        r"^day$",
        r"^timestamp$",
        r"^time$",
        r"^period$",
        r"^month$",
        r"^year$",
    ]
    for p in time_patterns:
        for c in cols:
            if re.search(p, str(c).strip().lower()):
                if df[c].nunique(dropna=True) >= 3:
                    return c

    # 3. String date formats (e.g. YYYY-MM-DD or YYYY-Wxx)
    for c in cols:
        if df[c].dtype == object:
            sample_val = str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""
            if re.match(r"^\d{4}-\d{2}-\d{2}", sample_val) or re.match(r"^\d{4}-W\d{2}", sample_val):
                return c

    return None


def detect_slice_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Detects categorical slicing dimensions for SRM slice diagnostics (e.g. device, browser, platform).
    """
    if df.empty:
        return []
    exclude_cols = exclude_cols or []
    candidates = []

    preferred_slice_names = [
        r"device",
        r"browser",
        r"platform",
        r"country",
        r"os",
        r"operating_system",
        r"network",
        r"channel",
        r"category",
        r"tier",
        r"segment",
        r"region",
    ]

    for c in df.columns:
        if c in exclude_cols:
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 30 and not np.issubdtype(df[c].dtype, np.number):
            # Prioritize matching names
            if any(re.search(p, str(c).lower()) for p in preferred_slice_names):
                candidates.insert(0, c)
            else:
                candidates.append(c)

    # Remove duplicates preserving order
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def aggregate_user_data_to_panel(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregates granular user-level transactions or events into a balanced/unbalanced panel dataset.
    This enables users who upload raw user logs (with e.g. date and country/platform) to instantly
    run regional Difference-in-Differences and Synthetic Control models!
    """
    if df.empty or unit_col not in df.columns or time_col not in df.columns or outcome_col not in df.columns:
        return pd.DataFrame()

    group_cols = [unit_col, time_col]
    agg_dict: Dict[str, Any] = {
        outcome_col: "mean",
    }
    if treatment_col and treatment_col in df.columns:
        # If binary treatment, max or first
        agg_dict[treatment_col] = "max"

    panel_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    panel_df = panel_df.rename(columns={outcome_col: f"avg_{outcome_col}"})
    return panel_df

