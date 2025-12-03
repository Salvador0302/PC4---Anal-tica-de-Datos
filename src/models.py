from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


# =========================
# Custom Losses
# =========================


class WeightedBinaryCrossentropy(keras.losses.Loss):
    """BCE ponderada para manejar desbalance de clases.

    loss = - w_pos * y * log(p) - w_neg * (1 - y) * log(1 - p)
    """

    def __init__(self, pos_weight: float = 2.0, name: str = "weighted_bce"):
        super().__init__(name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss_pos = -self.pos_weight * y_true * tf.math.log(y_pred)
        loss_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"pos_weight": self.pos_weight})
        return config


class FocalLoss(keras.losses.Loss):
    """Focal Loss simplificada para clasificación binaria.

    Referencia: Lin et al. 2017
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, name: str = "focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -(
            y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        loss = alpha_t * tf.pow(1.0 - p_t, self.gamma) * cross_entropy
        return tf.reduce_mean(loss)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


# Para modo regresión (conteo) podemos usar directamente keras.losses.Huber


# =========================
# Custom Metric: Recall@K Hotspots
# =========================


class RecallAtKHotspots(keras.metrics.Metric):
    """Métrica streaming simplificada de Recall@K.

    Versión por batch: asume que cada batch representa "una foto" del riesgo.

    Limitación importante (explicada en README y comentarios):
    - Idealmente se debería agrupar por semana completa antes de rankear.
    - Aquí, por simplicidad, consideramos el batch como conjunto de distritos.
    """

    def __init__(self, k: int = config.DEFAULT_K_HOTSPOTS, name: str = "recall_at_k", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total_true_positives = self.add_weight(name="total_tp", initializer="zeros")
        self.total_positives = self.add_weight(name="total_pos", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # Ordenar por score descendente y tomar top-k
        k = tf.minimum(self.k, tf.size(y_pred))
        _, topk_indices = tf.nn.top_k(y_pred, k=k, sorted=False)

        y_true_topk = tf.gather(y_true, topk_indices)
        tp = tf.reduce_sum(y_true_topk)
        pos = tf.reduce_sum(y_true)

        self.total_true_positives.assign_add(tp)
        self.total_positives.assign_add(pos)

    def result(self):
        return tf.math.divide_no_nan(self.total_true_positives, self.total_positives)

    def reset_states(self):
        self.total_true_positives.assign(0.0)
        self.total_positives.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"k": self.k})
        return cfg


# =========================
# Custom Layer
# =========================


class MyDense(layers.Layer):
    """Capa densa custom simple.

    Implementa: y = activation(x @ W + b)
    """

    def __init__(self, units: int, activation: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias", shape=(self.units,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        z = tf.linalg.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            return self.activation(z)
        return z

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"units": self.units, "activation": self.activation_name})
        return cfg


# =========================
# Custom Model
# =========================


class RiskModel(keras.Model):
    """Modelo básico de riesgo usando MyDense.

    Incluye ejemplo con add_loss para regularización L2 manual.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units=None,
        dropout_rate: float = config.DEFAULT_DROPOUT,
        task: str = "classification",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_units is None:
            hidden_units = config.DEFAULT_HIDDEN_UNITS

        self.task = task
        self.hidden_layers = []
        for i, u in enumerate(hidden_units):
            self.hidden_layers.append(MyDense(u, activation="relu", name=f"mydense_{i}"))
        self.dropout = layers.Dropout(dropout_rate)

        if task == "classification":
            self.output_layer = MyDense(1, activation="sigmoid", name="risk_score")
        elif task == "poisson":
            # Para Poisson usamos salida positiva con softplus
            self.output_layer = MyDense(1, activation="softplus", name="risk_rate")
        else:
            raise ValueError("task debe ser 'classification' o 'poisson'")

        self.reg_lambda = 1e-4

    def call(self, inputs, training=False):
        x = inputs
        l2_reg = 0.0
        for layer in self.hidden_layers:
            x = layer(x)
            # Regularización L2 manual sobre los kernels
            if hasattr(layer, "W"):
                l2_reg += tf.reduce_sum(tf.square(layer.W))
        if training:
            x = self.dropout(x, training=training)
        outputs = self.output_layer(x)

        # add_loss para incluir regularización en el total
        self.add_loss(self.reg_lambda * l2_reg)
        return outputs

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "task": self.task,
                "hidden_units": [l.units for l in self.hidden_layers],
                "dropout_rate": self.dropout.rate,
                "reg_lambda": self.reg_lambda,
            }
        )
        return cfg


def build_risk_model(input_dim: int, task: str = "classification", **kwargs) -> RiskModel:
    return RiskModel(input_dim=input_dim, task=task, **kwargs)


def get_loss(task: str = "classification", variant: str = "weighted_bce"):
    if task == "classification":
        if variant == "weighted_bce":
            return WeightedBinaryCrossentropy(pos_weight=2.0)
        if variant == "focal":
            return FocalLoss(gamma=2.0, alpha=0.25)
        if variant == "bce":
            return keras.losses.BinaryCrossentropy()
        if variant == "bce_ls":
            return keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        if variant == "bce_pos_weight_3":
            return WeightedBinaryCrossentropy(pos_weight=3.0)
        raise ValueError(f"Loss variante desconocida: {variant}")
    elif task == "poisson":
        if variant == "huber":
            return keras.losses.Huber()
        if variant == "poisson":
            return keras.losses.Poisson()
        raise ValueError(f"Loss variante desconocida para Poisson: {variant}")
    else:
        raise ValueError("task debe ser 'classification' o 'poisson'")


def get_metrics(task: str = "classification", use_custom_recall: bool = True):
    metrics = []
    if task == "classification":
        metrics.append(keras.metrics.AUC(curve="PR", name="pr_auc"))
        metrics.append(keras.metrics.Precision(name="precision"))
        metrics.append(keras.metrics.Recall(name="recall"))
        if use_custom_recall:
            metrics.append(RecallAtKHotspots(k=config.DEFAULT_K_HOTSPOTS))
    else:
        metrics.append(keras.metrics.MeanAbsoluteError(name="mae"))
    return metrics


def get_compile_kwargs(task: str = "classification", loss_variant: str = "weighted_bce", use_custom_recall: bool = True):
    return {
        "loss": get_loss(task=task, variant=loss_variant),
        "metrics": get_metrics(task=task, use_custom_recall=use_custom_recall),
    }
