"""Configuración compartida por la app, el modelo de visión y el agente."""

from os import getenv
from pathlib import Path

from dotenv import load_dotenv


RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_ASSETS = RAIZ / "assets"
DIR_FALLOS = DIR_ASSETS / "fallos"
DIR_EJEMPLOS = DIR_ASSETS / "ejemplos"
RUTA_INFO_MODELO = DIR_MODELOS / "modelo_info.json"

TAMANO_IMAGEN = 160
CLASE_POSITIVA = "ocupada"
CLASE_NEGATIVA = "vacía"
UMBRAL_DECISION = 0.5

MODELO_GEMINI = getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
