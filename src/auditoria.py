"""Auditoría reproducible de los pesos incluidos, sin reentrenar ni usar Gemini."""
import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.modelo import RAIZ, PESOS, cargar_modelo, imagen_dataset, preparar, predecir


def auditar():
    datos = load_digits()
    indices = np.arange(len(datos.target))
    desarrollo, prueba = train_test_split(indices, test_size=0.15, random_state=41,
                                         stratify=datos.target)
    entreno, validacion = train_test_split(desarrollo, test_size=0.15 / 0.85,
        random_state=41, stratify=datos.target[desarrollo])
    conjuntos = [set(entreno), set(validacion), set(prueba)]
    assert not any(conjuntos[i] & conjuntos[j] for i in range(3) for j in range(i + 1, 3))
    assert set.union(*conjuntos) == set(indices)
    entradas = np.stack([preparar(imagen_dataset(datos.images[i])) for i in prueba])
    probabilidades = np.asarray(cargar_modelo()(entradas, training=False))
    clases = probabilidades.argmax(axis=1)
    confianza = probabilidades.max(axis=1) * 100
    aciertos = clases == datos.target[prueba]
    errores = [{"indice": int(prueba[i]), "real": int(datos.target[prueba[i]]),
                "predicho": int(clases[i]), "confianza_pct": round(float(confianza[i]), 2)}
               for i in np.flatnonzero(~aciertos)]
    original = np.asarray(Image.open(RAIZ / "assets/caso_prueba.png").convert("L"))
    caso = json.loads((RAIZ / "assets/caso_prueba.json").read_text(encoding="utf-8"))
    assert np.array_equal(original, imagen_dataset(datos.images[caso["indice"]]))
    assert predecir(original)["digito"] == caso["digito"]
    assert abs(predecir(original)["confianza_pct"] - caso["confianza_pct"]) <= 0.02
    cambios = []
    for nombre, tamano, metodo in [("original", 8, Image.Resampling.NEAREST),
                                  ("vecino_mas_cercano", 192, Image.Resampling.NEAREST),
                                  ("vecino_mas_cercano", 256, Image.Resampling.NEAREST),
                                  ("bicubica", 256, Image.Resampling.BICUBIC)]:
        imagen = np.asarray(Image.fromarray(original).resize((tamano, tamano), metodo))
        cambios.append({"interpolacion": nombre, "tamano": tamano, **predecir(imagen)})
    resultado = {
        "alcance": "Auditoría de inferencia sobre los pesos exportados por U4.",
        "python": platform.python_version(),
        "versiones": {p: version(p) for p in ["keras", "jax", "numpy", "scikit-learn", "pillow"]},
        "sha256_modelo": hashlib.sha256(PESOS.read_bytes()).hexdigest(),
        "particiones": {"entrenamiento": len(entreno), "validacion": len(validacion), "prueba": len(prueba)},
        "particiones_disjuntas": True,
        "aciertos": int(aciertos.sum()), "total": len(prueba), "exactitud": float(aciertos.mean()),
        "errores": errores, "lecturas_bajo_90": int((confianza < 90).sum()),
        "errores_bajo_90": int(((confianza < 90) & ~aciertos).sum()),
        "aciertos_bajo_90": int(((confianza < 90) & aciertos).sum()),
        "interpolacion_caso": cambios,
    }
    (RAIZ / "assets/auditoria_modelo.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado


if __name__ == "__main__":
    auditar()
