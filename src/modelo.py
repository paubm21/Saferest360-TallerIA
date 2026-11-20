"""La interfaz y el entrenamiento comparten la misma transformación de píxeles."""
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import cv2
import keras
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PESOS = RAIZ / "models" / "lector.keras"


def imagen_dataset(matriz):
    """Convierte niveles 0..16 de load_digits en una imagen PNG de 8 bits."""
    return np.rint(matriz * (255 / 16)).astype(np.uint8)


def preparar(imagen):
    """Normaliza un dígito recortado con fondo claro u oscuro a 8 × 8 × 1."""
    gris = np.asarray(imagen)
    if gris.ndim == 3:
        gris = cv2.cvtColor(gris, cv2.COLOR_RGB2GRAY)
    if gris.ndim != 2 or gris.size == 0 or gris.dtype != np.uint8:
        raise ValueError("La lectura necesita una imagen de 8 bits.")
    esquinas = gris[[0, 0, -1, -1], [0, -1, 0, -1]]
    if np.median(esquinas) > 127:
        gris = 255 - gris
    reducida = cv2.resize(gris, (8, 8), interpolation=cv2.INTER_AREA)
    if reducida.max() == reducida.min():
        raise ValueError("La imagen es uniforme. Sube un recorte con un solo dígito.")
    return (reducida.astype("float32") / 255)[..., None]


@lru_cache(maxsize=1)
def cargar_modelo():
    return keras.models.load_model(PESOS, compile=False)


def predecir(imagen):
    vector = np.asarray(cargar_modelo()(preparar(imagen)[None], training=False))[0]
    digito = int(vector.argmax())
    return {"digito": digito, "confianza_pct": round(float(vector[digito]) * 100, 2)}
