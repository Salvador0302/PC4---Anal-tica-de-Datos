"""Ejercicio 1 - Capítulo 12

Custom Layer + Custom Loss + serialización.

Este script es autocontenido y didáctico.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


# 1) Definimos una capa custom muy simple
class SimpleScaleLayer(layers.Layer):
    """Escala la entrada por un factor entrenable.

    Punto difícil 1: usar add_weight en build() para crear parámetros.
    """

    def __init__(self, initial_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"initial_scale": self.initial_scale})
        return cfg


# 2) Definimos una loss custom muy simple
class MeanSquaredScaledError(keras.losses.Loss):
    """MSE con un factor global.

    Punto difícil 2: heredar de keras.losses.Loss y usar get_config.
    """

    def __init__(self, factor: float = 1.0, name: str = "msse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = factor

    def call(self, y_true, y_pred):
        return self.factor * tf.reduce_mean(tf.square(y_true - y_pred))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"factor": self.factor})
        return cfg


def build_toy_model():
    """Modelo pequeño que usa la capa y la loss custom.

    Punto difícil 3: integrar una capa custom dentro de un modelo secuencial.
    """

    inputs = keras.Input(shape=(1,), name="x")
    x = SimpleScaleLayer(initial_scale=0.5, name="scale_layer")(inputs)
    outputs = layers.Dense(1, name="dense_out")(x)

    model = keras.Model(inputs, outputs, name="toy_scale_model")
    model.compile(optimizer="adam", loss=MeanSquaredScaledError(factor=2.0))
    return model


def run_training_and_save():
    # Datos sintéticos simples: y = 3x + ruido
    rng = np.random.default_rng(config.RANDOM_SEED)
    x = rng.uniform(-1, 1, size=(200, 1)).astype("float32")
    noise = rng.normal(0, 0.1, size=(200, 1)).astype("float32")
    y = 3 * x + noise

    model = build_toy_model()
    model.fit(x, y, epochs=10, batch_size=32, verbose=0)

    out_dir = os.path.join(config.REPORTS_DIR, "ejercicio1_model")
    os.makedirs(out_dir, exist_ok=True)

    # Punto difícil 4: guardar el modelo con componentes custom.
    model.save(out_dir)

    return out_dir


def load_and_test_model(model_dir: str):
    """Recarga el modelo y verifica que produce salidas razonables.

    Punto difícil 5: pasar custom_objects a load_model.
    """

    loaded = keras.models.load_model(
        model_dir,
        custom_objects={
            "SimpleScaleLayer": SimpleScaleLayer,
            "MeanSquaredScaledError": MeanSquaredScaledError,
        },
    )

    test_x = np.array([[0.5]], dtype="float32")
    pred = loaded.predict(test_x, verbose=0)
    print(f"Predicción recargada para x=0.5: {pred[0,0]:.4f}")


def main():
    tf.random.set_seed(config.RANDOM_SEED)
    model_dir = run_training_and_save()
    load_and_test_model(model_dir)


if __name__ == "__main__":
    main()
