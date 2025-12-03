"""Script para correr al menos 20 experimentos de modelos.

Genera una tabla comparativa en reports/leaderboard.csv y selecciona el mejor
modelo según PR-AUC (si aplica).
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config, data, models


@dataclass
class ExperimentConfig:
    name: str
    loss_variant: str
    use_custom_recall: bool
    hidden_units: List[int]
    dropout: float
    training_mode: str  # "fit" o "custom_loop"
    use_tf_function: bool


@dataclass
class ExperimentResult:
    name: str
    pr_auc: float
    recall_at_k: float
    f1: float
    loss: float
    train_time_sec: float
    params: Dict


def build_experiments() -> List[ExperimentConfig]:
    exps: List[ExperimentConfig] = []

    # 5 variantes de loss
    loss_variants = [
        "weighted_bce",
        "focal",
        "bce",
        "bce_ls",
        "bce_pos_weight_3",
    ]
    for lv in loss_variants:
        exps.append(
            ExperimentConfig(
                name=f"loss_{lv}",
                loss_variant=lv,
                use_custom_recall=True,
                hidden_units=config.DEFAULT_HIDDEN_UNITS,
                dropout=config.DEFAULT_DROPOUT,
                training_mode="fit",
                use_tf_function=True,
            )
        )

    # 5 variantes de métricas (activar/desactivar custom recall)
    for i, use_cr in enumerate([True, False] * 3)[:5]:
        exps.append(
            ExperimentConfig(
                name=f"metrics_customrecall_{i}_use_{int(use_cr)}",
                loss_variant="weighted_bce",
                use_custom_recall=use_cr,
                hidden_units=config.DEFAULT_HIDDEN_UNITS,
                dropout=config.DEFAULT_DROPOUT,
                training_mode="fit",
                use_tf_function=True,
            )
        )

    # 5 variantes de arquitectura usando MyDense
    archs = [
        [16],
        [64, 32],
        [64, 32, 16],
        [128],
        [32, 32],
    ]
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]
    for i in range(5):
        exps.append(
            ExperimentConfig(
                name=f"arch_{i}",
                loss_variant="weighted_bce",
                use_custom_recall=True,
                hidden_units=archs[i],
                dropout=dropouts[i],
                training_mode="fit",
                use_tf_function=True,
            )
        )

    # 5 variantes de entrenamiento (fit vs custom loop, tf.function on/off)
    modes = [
        ("fit", True),
        ("fit", False),
        ("custom_loop", True),
        ("custom_loop", False),
        ("custom_loop", True),
    ]
    for i, (tm, tf_f) in enumerate(modes):
        exps.append(
            ExperimentConfig(
                name=f"trainmode_{i}_{tm}_tff_{int(tf_f)}",
                loss_variant="weighted_bce",
                use_custom_recall=True,
                hidden_units=config.DEFAULT_HIDDEN_UNITS,
                dropout=config.DEFAULT_DROPOUT,
                training_mode=tm,
                use_tf_function=tf_f,
            )
        )

    assert len(exps) >= 20
    return exps


def run_experiment(
    exp: ExperimentConfig,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) -> ExperimentResult:
    input_dim = X_train.shape[1]

    model = models.RiskModel(
        input_dim=input_dim,
        hidden_units=exp.hidden_units,
        dropout_rate=exp.dropout,
        task="classification",
    )

    compile_kwargs = models.get_compile_kwargs(
        task="classification", loss_variant=exp.loss_variant, use_custom_recall=exp.use_custom_recall
    )

    loss_fn = compile_kwargs["loss"]
    metric_objs = compile_kwargs["metrics"]

    optimizer = keras.optimizers.Adam(1e-3)

    batch_size = 64
    epochs = 4

    start = time.time()

    if exp.training_mode == "fit":
        model.compile(optimizer=optimizer, **compile_kwargs)
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
    else:
        # custom_loop sencillo (solo train, sin tf.function si exp.use_tf_function=False)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        if exp.use_tf_function:
            @tf.function
            def train_step(xb, yb):
                with tf.GradientTape() as tape:
                    logits = model(xb, training=True)
                    logits = tf.reshape(logits, tf.shape(yb))
                    loss_value = loss_fn(yb, logits)
                    if model.losses:
                        loss_value += tf.add_n(model.losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                for m in metric_objs:
                    m.update_state(yb, logits)
                return loss_value
        else:

            def train_step(xb, yb):
                with tf.GradientTape() as tape:
                    logits = model(xb, training=True)
                    logits = tf.reshape(logits, tf.shape(yb))
                    loss_value = loss_fn(yb, logits)
                    if model.losses:
                        loss_value += tf.add_n(model.losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                for m in metric_objs:
                    m.update_state(yb, logits)
                return loss_value

        for epoch in range(epochs):
            for m in metric_objs:
                m.reset_states()
            for xb, yb in train_ds:
                _ = train_step(xb, yb)

    train_time = time.time() - start

    # Evaluar en test
    y_scores = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
    y_true = y_test.reshape(-1)

    # PR-AUC manual usando keras.metrics
    pr_auc_metric = keras.metrics.AUC(curve="PR")
    pr_auc_metric.update_state(y_true, y_scores)
    pr_auc = float(pr_auc_metric.result().numpy())

    # Recall@K manual sencillo
    k = min(config.DEFAULT_K_HOTSPOTS, len(y_scores))
    topk_idx = np.argsort(-y_scores)[:k]
    recall_at_k = float(y_true[topk_idx].sum() / max(y_true.sum(), 1.0))

    # F1 binario simple usando umbral 0.5
    y_pred_bin = (y_scores >= 0.5).astype(int)
    tp = float(((y_pred_bin == 1) & (y_true == 1)).sum())
    fp = float(((y_pred_bin == 1) & (y_true == 0)).sum())
    fn = float(((y_pred_bin == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    loss_metric = keras.losses.BinaryCrossentropy()
    loss_value = float(loss_metric(y_true, y_scores).numpy())

    params = asdict(exp)

    return ExperimentResult(
        name=exp.name,
        pr_auc=pr_auc,
        recall_at_k=recall_at_k,
        f1=f1,
        loss=loss_value,
        train_time_sec=train_time,
        params=params,
    )


def main():
    tf.random.set_seed(config.RANDOM_SEED)

    df_raw = data.load_csv(config.DEFAULT_DATA_PATH)
    panel = data.make_panel_district_week(df_raw)
    train_df, val_df, test_df = data.train_val_test_split_time(panel)

    X_train, y_train, _ = data.get_feature_target_arrays(train_df, task="classification")
    X_val, y_val, _ = data.get_feature_target_arrays(val_df, task="classification")
    X_test, y_test, _ = data.get_feature_target_arrays(test_df, task="classification")

    exps = build_experiments()

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    results: List[ExperimentResult] = []

    for i, exp in enumerate(exps):
        print(f"=== Experimento {i+1}/{len(exps)}: {exp.name} ===")
        res = run_experiment(exp, X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(res)

    # Construir leaderboard
    rows = []
    for r in results:
        row = {
            "name": r.name,
            "pr_auc": r.pr_auc,
            "recall_at_k": r.recall_at_k,
            "f1": r.f1,
            "loss": r.loss,
            "train_time_sec": r.train_time_sec,
        }
        row.update({f"param_{k}": v for k, v in r.params.items()})
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    leaderboard_path = os.path.join(config.REPORTS_DIR, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)

    # Guardar mejor modelo
    best_name = leaderboard.iloc[0]["name"]
    print(f"Mejor experimento: {best_name}")

    # (Re-entrenar mejor modelo rápido y guardar)
    best_exp = next(e for e in exps if e.name == best_name)
    input_dim = X_train.shape[1]
    best_model = models.RiskModel(
        input_dim=input_dim,
        hidden_units=best_exp.hidden_units,
        dropout_rate=best_exp.dropout,
        task="classification",
    )
    compile_kwargs = models.get_compile_kwargs(
        task="classification",
        loss_variant=best_exp.loss_variant,
        use_custom_recall=best_exp.use_custom_recall,
    )
    best_model.compile(optimizer=keras.optimizers.Adam(1e-3), **compile_kwargs)
    best_model.fit(
        np.concatenate([X_train, X_val], axis=0),
        np.concatenate([y_train, y_val], axis=0),
        epochs=6,
        batch_size=64,
        verbose=0,
    )
    best_dir = os.path.join(config.REPORTS_DIR, "best_experiment_model")
    os.makedirs(best_dir, exist_ok=True)
    best_model.save(best_dir)

    # Plot comparativo
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(leaderboard["name"], leaderboard["pr_auc"])
    ax.set_xticklabels(leaderboard["name"], rotation=90)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Comparación de 20+ modelos")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "leaderboard_pr_auc.png"))
    plt.close(fig)

    print(f"Leaderboard guardado en {leaderboard_path}")


if __name__ == "__main__":
    main()
