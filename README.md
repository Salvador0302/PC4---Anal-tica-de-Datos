# Sistema de alerta temprana de extorsión (Capítulo 12 - TensorFlow/Keras)

Este proyecto implementa, en Python y TensorFlow (Keras), un sistema **simple y ejecutable en CPU** para estimar el riesgo de que ocurra **al menos 1 caso de extorsión (u otro delito)** por **distrito-semana** en Perú. El enfoque sigue las ideas del **Capítulo 12: Custom Models and Training with TensorFlow (Keras)**:

- Pérdidas custom (custom losses)
- Métricas custom stateful
- Capas custom (subclase de `Layer`)
- Modelos custom (subclase de `Model`)
- Entrenamiento manual con `tf.GradientTape` y `tf.function`

Todo está pensado para correr en un **Codespace** con CPU.

---

## 1. Estructura del repositorio

```text
PC4---Anal-tica-de-Datos/
├─ data/
│  └─ crime_demo.csv        # (opcional) tu CSV real; si no existe, se genera un dataset sintético
├─ src/
│  ├─ config.py             # rutas, columnas esperadas, semillas y parámetros por defecto
│  ├─ data.py               # carga de CSV, panel distrito-semana, lags, splits
│  ├─ models.py             # custom losses, métricas, capa MyDense y modelo RiskModel
│  ├─ train.py              # entrenamiento base con model.fit()
│  ├─ custom_loop.py        # entrenamiento manual con GradientTape + tf.function
│  ├─ experiments_20_models.py
│  │                        # corrida de al menos 20 modelos y leaderboard
│  ├─ EJERCICIO_1_Grupo_2_CAPITULO_12.py  # custom layer + custom loss + serialización
│  └─ EJERCICIO_2_Grupo_2_CAPITULO_12.py  # custom training loop con métricas streaming
├─ reports/
│  ├─ ...                   # plots, métricas y modelos guardados
└─ [requirements.txt](http://_vscodecontentref_/2)