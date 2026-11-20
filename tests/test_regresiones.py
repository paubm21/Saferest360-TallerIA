"""Regresiones locales. No sustituyen la evaluación en vivo del notebook U5."""
import ast
import json
import os
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.agente import aplicar_regla, decision
from src.modelo import RAIZ, preparar, predecir


class Regresiones(unittest.TestCase):
    def test_limites_y_escalamiento(self):
        casos = [(0, "REVISAR", "SEGUNDA_LECTURA"),
                 (57.95, "REVISAR", "SEGUNDA_LECTURA"),
                 (59.99, "REVISAR", "SEGUNDA_LECTURA"),
                 (60, "REVISAR", "REVISION_SIMPLE"),
                 (89.99, "REVISAR", "REVISION_SIMPLE"),
                 (90, "ACEPTAR", "SIN_ESCALAMIENTO"),
                 (100, "ACEPTAR", "SIN_ESCALAMIENTO")]
        for confianza, accion, nivel in casos:
            with self.subTest(confianza=confianza):
                r = json.loads(aplicar_regla.invoke({"confianza_pct": confianza}))
                self.assertEqual(r["decision"], accion)
                self.assertEqual(r["escalamiento"], nivel)

    def test_confianza_invalida(self):
        for valor in [-1, 101, float("nan"), float("inf")]:
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                decision(valor)

    def test_entrada_uniforme_se_rechaza(self):
        for valor in [0, 255]:
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                preparar(np.full((8, 8), valor, dtype=np.uint8))

    def test_caso_exportado_y_ejemplo_app(self):
        from app import ejemplo, lectura
        caso = json.loads((RAIZ / "assets/caso_prueba.json").read_text(encoding="utf-8"))
        original = np.asarray(Image.open(RAIZ / "assets/caso_prueba.png").convert("L"))
        salida = predecir(original)
        self.assertEqual(salida["digito"], caso["digito"])
        self.assertAlmostEqual(salida["confianza_pct"], caso["confianza_pct"], delta=0.02)
        self.assertEqual(predecir(ejemplo()), salida)
        self.assertIn(str(salida["digito"]), lectura(ejemplo()))

    @unittest.skipUnless(os.getenv("TALLER_TEST_UI") == "1",
                         "La construcción de Gradio necesita sockets locales: habilitar TALLER_TEST_UI=1 fuera del entorno restringido.")
    def test_construccion_interfaz(self):
        from app import interfaz
        self.assertIsNotNone(interfaz())

    def test_validador_original_no_relajado(self):
        antes = (RAIZ / "tests/validador_original.txt").read_text(encoding="utf-8")
        actual = json.loads((RAIZ / "notebooks/evaluacion_agente_u5.ipynb").read_text(encoding="utf-8"))
        def funciones(fuentes):
            return {
                nodo.name: ast.dump(nodo, include_attributes=False)
                for fuente in fuentes
                for nodo in ast.parse(fuente).body
                if isinstance(nodo, ast.FunctionDef) and nodo.name in {"normalizar", "resultado_correcto"}
            }

        originales = funciones([antes])
        self.assertEqual(set(originales), {"normalizar", "resultado_correcto"})
        celdas = ("".join(c["source"]) for c in actual["cells"] if c["cell_type"] == "code")
        self.assertEqual(originales, funciones(celdas))


if __name__ == "__main__":
    unittest.main()
