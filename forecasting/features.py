"""Feature engineering for venue demand forecasting."""
import pandas as pd
import numpy as np


SCHOOL_HOLIDAY_MONTHS = [6, 7, 8, 12]  # Jun, Jul, Aug, Dec
PEAK_MONTHS = [6, 7, 8]
LOCAL_EVENT_DATES = []  # Populated from ROLLER API in production


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Calendar features
    df["day_of_week"]     = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"]      = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"]           = df["date"].dt.month
    df["day_of_month"]    = df["date"].dt.day
    df["week_of_year"]    = df["date"].dt.isocalendar().week.astype(int)
    df["is_school_holiday"] = df["month"].isin(SCHOOL_HOLIDAY_MONTHS).astype(int)
    df["is_peak_season"]  = df["month"].isin(PEAK_MONTHS).astype(int)

    # Cyclical encoding for day of week and month (preserves circular structure)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Holiday flags (US federal holidays approximated)
    df["is_holiday"] = 0
    holidays = ["01-01", "07-04", "11-25", "12-25", "12-31"]  # approx
    for h in holidays:
        mask = df["date"].dt.strftime("%m-%d") == h
        df.loc[mask, "is_holiday"] = 1

    feature_cols = [
        "day_of_week", "is_weekend", "month", "day_of_month", "week_of_year",
        "is_school_holiday", "is_peak_season", "is_holiday",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    return df[feature_cols]
