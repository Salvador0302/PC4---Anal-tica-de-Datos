import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_csv(path: str) -> pd.DataFrame:
    """Carga un CSV y hace validaciones básicas.

    Si el archivo no existe, genera un dataset sintético de ejemplo.
    """
    if not os.path.exists(path):
        print(f"[data] Archivo {path} no existe. Generando datos sintéticos de ejemplo...")
        df = generate_synthetic_data()
        return df

    df = pd.read_csv(path)
    return df


def generate_synthetic_data(n_districts: int = 20, n_weeks: int = 80) -> pd.DataFrame:
    """Genera un panel sintético distrito-semana para demo.

    Columnas: district, date (YYYY-MM-DD), count
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    districts = [f"DIST_{i:02d}" for i in range(1, n_districts + 1)]
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

    rows = []
    for d in districts:
        base_rate = rng.uniform(0.1, 1.0)
        for dt in dates:
            lam = base_rate * rng.uniform(0.5, 2.0)
            cnt = rng.poisson(lam)
            rows.append({"district": d, "date": dt.date().isoformat(), "count": int(cnt)})

    df = pd.DataFrame(rows)
    return df


def make_panel_district_week(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma el dataset a panel distrito-semana.

    - Requiere al menos: district, date, count.
    - Agrupa por semana ISO (usamos inicio de semana como lunes).
    - Suma conteos por distrito-semana.
    - Crea target binario y (opcional) target de conteo t+1.
    - Crea features de lags y rolling mean.
    """
    for col in config.REQUIRED_COLUMNS_MIN:
        if col not in df.columns:
            raise ValueError(f"Columna requerida faltante: {col}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Definir semana como el lunes de cada semana
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time

    group_cols = ["district", "week_start"]
    agg_dict = {"count": "sum"}

    # Mantener features extra agregadas por suma o media simple
    extra_cols = [c for c in df.columns if c not in ["district", "date", "count", "week_start"]]
    for c in extra_cols:
        agg_dict[c] = "mean"

    panel = df.groupby(group_cols, as_index=False).agg(agg_dict).sort_values(["district", "week_start"])

    # Ordenar y crear índices por distrito
    panel = panel.sort_values(["district", "week_start"]).reset_index(drop=True)

    # Crear lags y rolling por distrito
    def _add_lags(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("week_start").reset_index(drop=True)
        for l in config.LAG_STEPS:
            g[f"count_lag{l}"] = g["count"].shift(l)
        g["count_roll_mean4"] = g["count"].rolling(config.ROLLING_WINDOW, min_periods=1).mean()
        return g

    panel = panel.groupby("district", group_keys=False).apply(_add_lags)

    # Crear target binario y target de conteo t+1 (para modo Poisson)
    panel["target_count_t1"] = panel.groupby("district")["count"].shift(-1)
    panel["target_bin_t1"] = (panel["target_count_t1"] >= 1).astype(float)

    # Eliminar filas sin target (última semana por distrito)
    panel = panel.dropna(subset=["target_count_t1"]).reset_index(drop=True)

    return panel


def train_val_test_split_time(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal simple respetando el orden.

    Se asume que df ya está ordenado por tiempo.
    """
    df = df.sort_values(["week_start"]).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    n_val = int((n - n_test) * val_size)

    train = df.iloc[: n - n_test - n_val]
    val = df.iloc[n - n_test - n_val : n - n_test]
    test = df.iloc[n - n_test :]

    return train, val, test


def get_feature_target_arrays(
    df: pd.DataFrame,
    task: str = "classification",
):
    """Devuelve X, y según el modo de tarea.

    - Features: lags, rolling y cualquier columna numérica adicional.
    - task="classification": y = target_bin_t1
    - task="poisson": y = target_count_t1
    """
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "district",
            "date",
            "week_start",
            "target_bin_t1",
            "target_count_t1",
        ]
        and np.issubdtype(df[c].dtype, np.number)
    ]

    if task == "classification":
        target_col = "target_bin_t1"
    elif task == "poisson":
        target_col = "target_count_t1"
    else:
        raise ValueError("task debe ser 'classification' o 'poisson'")

    X = df[feature_cols].values.astype("float32")
    y = df[target_col].values.astype("float32")

    return X, y, feature_cols
