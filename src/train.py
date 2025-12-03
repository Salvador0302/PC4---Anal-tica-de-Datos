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
    parser = argparse.ArgumentParser(description="Entrenamiento base de modelo de riesgo de extorsión")
    parser.add_argument("--data", dest="data_path", type=str, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--task", type=str, default=config.DEFAULT_TASK, choices=["classification", "poisson"])
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def ensure_reports_dirs():
    model_dir = os.path.join(config.REPORTS_DIR, "best_model")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    return model_dir


def main():
    args = parse_args()
    tf.random.set_seed(config.RANDOM_SEED)

    df_raw = data.load_csv(args.data_path)
    panel = data.make_panel_district_week(df_raw)

    train_df, val_df, test_df = data.train_val_test_split_time(panel)

    X_train, y_train, feature_cols = data.get_feature_target_arrays(train_df, task=args.task)
    X_val, y_val, _ = data.get_feature_target_arrays(val_df, task=args.task)
    X_test, y_test, _ = data.get_feature_target_arrays(test_df, task=args.task)

    input_dim = X_train.shape[1]

    model = models.build_risk_model(input_dim=input_dim, task=args.task)
    compile_kwargs = models.get_compile_kwargs(task=args.task, loss_variant="weighted_bce")

    if args.task == "poisson":
        compile_kwargs = models.get_compile_kwargs(task=args.task, loss_variant="poisson")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), **compile_kwargs)

    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")]

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )
    elapsed = time.time() - start

    # Evaluar en test
    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    results["train_time_sec"] = elapsed

    model_dir = ensure_reports_dirs()
    model.save(model_dir)

    # Guardar métricas
    metrics_path = os.path.join(config.REPORTS_DIR, "metrics_base.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plots de loss y PR-AUC (si disponible)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history.history["loss"], label="loss_train")
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="loss_val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "loss_curve.png"))
    plt.close(fig)

    if "pr_auc" in history.history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history["pr_auc"], label="pr_auc_train")
        if "val_pr_auc" in history.history:
            ax.plot(history.history["val_pr_auc"], label="pr_auc_val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PR-AUC")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(config.REPORTS_DIR, "pr_auc_curve.png"))
        plt.close(fig)

    print("Entrenamiento finalizado. Resultados test:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
