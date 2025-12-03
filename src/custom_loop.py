"""Entrenamiento manual con GradientTape + tf.function.

Compara un entrenamiento manual con model.fit() usando el mismo modelo.
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config, data, models


def parse_args():
    parser = argparse.ArgumentParser(description="Custom training loop para riesgo de extorsión")
    parser.add_argument("--data", dest="data_path", type=str, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--task", type=str, default=config.DEFAULT_TASK, choices=["classification", "poisson"])
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def ensure_reports_dirs():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)


def run_fit_baseline(X_train, y_train, X_val, y_val, task: str):
    input_dim = X_train.shape[1]
    model = models.build_risk_model(input_dim=input_dim, task=task)
    compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="weighted_bce")
    if task == "poisson":
        compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="poisson")

    model.compile(optimizer=keras.optimizers.Adam(1e-3), **compile_kwargs)

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=64,
        verbose=0,
    )
    elapsed = time.time() - start

    return model, history.history, elapsed


def run_custom_loop(X_train, y_train, X_val, y_val, task: str, epochs: int, batch_size: int):
    input_dim = X_train.shape[1]
    model = models.build_risk_model(input_dim=input_dim, task=task)

    compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="weighted_bce")
    if task == "poisson":
        compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="poisson")

    loss_fn = compile_kwargs["loss"]
    metric_objs = compile_kwargs["metrics"]

    optimizer = keras.optimizers.Adam(1e-3)

    train_loss_results = []
    val_loss_results = []

    # Para registrar una métrica principal (ej. PR-AUC) en validation
    val_main_metric_name = metric_objs[0].name if metric_objs else "metric"
    val_main_metric_results = []

    # Dataset tf.data
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=config.RANDOM_SEED)
        .batch(batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            logits = tf.reshape(logits, tf.shape(y_batch))
            loss_value = loss_fn(y_batch, logits)
            if model.losses:
                loss_value += tf.add_n(model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        for m in metric_objs:
            m.update_state(y_batch, logits)
        return loss_value

    @tf.function
    def val_step(x_batch, y_batch):
        logits = model(x_batch, training=False)
        logits = tf.reshape(logits, tf.shape(y_batch))
        loss_value = loss_fn(y_batch, logits)
        if model.losses:
            loss_value += tf.add_n(model.losses)
        for m in metric_objs:
            m.update_state(y_batch, logits)
        return loss_value

    start = time.time()
    for epoch in range(epochs):
        for m in metric_objs:
            m.reset_states()

        # Entrenamiento
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x_batch, y_batch in train_ds:
            loss_value = train_step(x_batch, y_batch)
            epoch_loss_avg.update_state(loss_value)
        train_loss = float(epoch_loss_avg.result().numpy())
        train_loss_results.append(train_loss)

        # Validación
        for m in metric_objs:
            m.reset_states()
        val_loss_avg = tf.keras.metrics.Mean()
        for x_batch, y_batch in val_ds:
            loss_value = val_step(x_batch, y_batch)
            val_loss_avg.update_state(loss_value)
        val_loss = float(val_loss_avg.result().numpy())
        val_loss_results.append(val_loss)

        main_metric_val = None
        if metric_objs:
            main_metric_val = float(metric_objs[0].result().numpy())
        val_main_metric_results.append(main_metric_val)

        print(
            f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f} - val_loss={val_loss:.4f} - {val_main_metric_name}={main_metric_val}"
        )

    elapsed = time.time() - start

    history = {
        "loss": train_loss_results,
        f"val_loss": val_loss_results,
        f"val_{val_main_metric_name}": val_main_metric_results,
    }
    return model, history, elapsed


def main():
    args = parse_args()
    tf.random.set_seed(config.RANDOM_SEED)

    df_raw = data.load_csv(args.data_path)
    panel = data.make_panel_district_week(df_raw)

    train_df, val_df, _ = data.train_val_test_split_time(panel)

    X_train, y_train, _ = data.get_feature_target_arrays(train_df, task=args.task)
    X_val, y_val, _ = data.get_feature_target_arrays(val_df, task=args.task)

    ensure_reports_dirs()

    # Baseline con fit
    fit_model, fit_history, fit_time = run_fit_baseline(X_train, y_train, X_val, y_val, args.task)

    # Custom loop
    cl_model, cl_history, cl_time = run_custom_loop(
        X_train, y_train, X_val, y_val, args.task, epochs=args.epochs, batch_size=args.batch_size
    )

    # Guardar métricas comparativas
    out = {
        "fit": {"history": fit_history, "time_sec": fit_time},
        "custom_loop": {"history": cl_history, "time_sec": cl_time},
    }

    json_path = os.path.join(config.REPORTS_DIR, "custom_loop_comparison.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # Plot comparativo de loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fit_history["loss"], label="fit_loss")
    ax.plot(cl_history["loss"], label="custom_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "custom_vs_fit_loss.png"))
    plt.close(fig)

    print("Comparación guardada en reports/custom_loop_comparison.json")


if __name__ == "__main__":
    main()
