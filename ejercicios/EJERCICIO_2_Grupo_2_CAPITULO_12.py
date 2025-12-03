"""Ejercicio 2 - Capítulo 12

Custom training loop con GradientTape + tf.function + métricas streaming.

Script autocontenido y didáctico.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config
from .models import RecallAtKHotspots


def build_simple_classifier():
    """Pequeño MLP para clasificación binaria.

    Punto difícil 1: usar una arquitectura mínima pero estable.
    """

    inputs = keras.Input(shape=(4,), name="features")
    x = layers.Dense(8, activation="relu")(inputs)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="toy_classifier")
    return model


def generate_toy_data(n: int = 400):
    """Datos binarios sintéticos.

    Punto difícil 2: generar un dataset no trivial pero rápido.
    """

    rng = np.random.default_rng(config.RANDOM_SEED)
    X = rng.normal(0, 1, size=(n, 4)).astype("float32")
    # y = 1 si suma de primeras 2 features > umbral
    logits = X[:, 0] + 0.5 * X[:, 1]
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, size=n) < probs).astype("float32")
    return X, y


def run_custom_training():
    tf.random.set_seed(config.RANDOM_SEED)
    X, y = generate_toy_data()

    # Split simple
    n = len(X)
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = build_simple_classifier()
    loss_fn = keras.losses.BinaryCrossentropy()

    # Métricas streaming
    pr_auc = keras.metrics.AUC(curve="PR", name="pr_auc")
    recall_at_k = RecallAtKHotspots(k=10)

    optimizer = keras.optimizers.Adam(1e-3)

    batch_size = 32
    epochs = 8

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(400).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    @tf.function
    def train_step(xb, yb):
        """Un paso de entrenamiento.

        Punto difícil 3: uso correcto de GradientTape y model.losses.
        """

        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            logits = tf.reshape(logits, tf.shape(yb))
            loss_value = loss_fn(yb, logits)
            if model.losses:
                loss_value += tf.add_n(model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        pr_auc.update_state(yb, logits)
        recall_at_k.update_state(yb, logits)
        return loss_value

    @tf.function
    def val_step(xb, yb):
        """Paso de validación.

        Punto difícil 4: reutilizar las mismas métricas en modo evaluación.
        """

        logits = model(xb, training=False)
        logits = tf.reshape(logits, tf.shape(yb))
        loss_value = loss_fn(yb, logits)
        if model.losses:
            loss_value += tf.add_n(model.losses)

        pr_auc.update_state(yb, logits)
        recall_at_k.update_state(yb, logits)
        return loss_value

    for epoch in range(epochs):
        # Reset métricas
        pr_auc.reset_states()
        recall_at_k.reset_states()

        # Entrenamiento
        for xb, yb in train_ds:
            loss_tr = train_step(xb, yb)

        train_pr_auc = float(pr_auc.result().numpy())
        train_recall_k = float(recall_at_k.result().numpy())

        # Validación
        pr_auc.reset_states()
        recall_at_k.reset_states()

        for xb, yb in val_ds:
            loss_v = val_step(xb, yb)

        val_pr_auc = float(pr_auc.result().numpy())
        val_recall_k = float(recall_at_k.result().numpy())

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss={loss_tr:.4f} pr_auc={train_pr_auc:.3f} recall@10={train_recall_k:.3f} - "
            f"val_loss={loss_v:.4f} val_pr_auc={val_pr_auc:.3f} val_recall@10={val_recall_k:.3f}"
        )

    # Punto difícil 5: asegurarse de que tf.function no rompe la métrica stateful.


def main():
    run_custom_training()


if __name__ == "__main__":
    main()
