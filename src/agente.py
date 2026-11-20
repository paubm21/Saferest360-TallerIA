"""Agente de inspección. La imagen viaja en el contexto local de cada turno."""
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from src.modelo import RAIZ, predecir

load_dotenv(RAIZ / ".env")
MODELO_PREDETERMINADO = "gemini-3.5-flash-lite"
UMBRAL = 90.0


@dataclass
class Contexto:
    imagen: np.ndarray | None


@tool
def leer_digito(runtime: ToolRuntime[Contexto]) -> str:
    """Lee el dígito de la imagen actual con la CNN y devuelve clase y confianza porcentual."""
    if runtime.context.imagen is None:
        return "Falta la imagen. Pide al usuario que suba un recorte."
    return json.dumps(predecir(runtime.context.imagen))


def decision(confianza_pct: float) -> str:
    if not 0 <= confianza_pct <= 100:
        raise ValueError("La confianza debe estar entre 0 y 100.")
    return "REVISAR" if confianza_pct < UMBRAL else "ACEPTAR"


@tool
def aplicar_regla(confianza_pct: float) -> str:
    """Aplica la política de revisión a la confianza de la CNN (0 a 100).

    Usa exactamente el porcentaje devuelto por leer_digito, sin convertirlo a 0..1.
    Por debajo de 60 pide segunda lectura independiente; de 60 a menos de 90
    pide revisión simple. A partir de 90 acepta provisionalmente.
    Devuelve decisión, escalamiento y motivo. No corrige el dígito de la CNN.
    """
    accion = decision(confianza_pct)
    if confianza_pct < 60:
        nivel, motivo = "SEGUNDA_LECTURA", "Confianza baja: dos lecturas humanas independientes; resolver discrepancias."
    elif confianza_pct < UMBRAL:
        nivel, motivo = "REVISION_SIMPLE", "Un operador debe confirmar o corregir la lectura."
    else:
        nivel, motivo = "SIN_ESCALAMIENTO", "Aceptación provisional; la confianza no garantiza acierto."
    return json.dumps({"decision": accion, "umbral_pct": UMBRAL,
                       "escalamiento": nivel, "motivo": motivo}, ensure_ascii=False)


INSTRUCCIONES = """Asistes a quien digitaliza formularios con un solo dígito por casilla.
Responde en español y brevemente. Para saludos o preguntas de memoria, conversa directamente.
Si se pide leer el dígito o indicar su confianza, usa únicamente leer_digito.
Termina ese turno con «Dígito: N. Confianza: P%», incluso cuando la confianza sea baja.
La regla se usa solo si el operador pide una decisión de aceptación, revisión o escalamiento.
Para decidir sobre una confianza explícita, llama a aplicar_regla con ese porcentaje.
Si piden leer y decidir sobre una imagen, usa primero leer_digito y luego aplicar_regla
con la confianza obtenida, en escala de 0 a 100 y sin modificarla.
La respuesta final de una inspección SIEMPRE debe incluir los TRES campos juntos:
«Dígito: N. Confianza: P%. Decisión: REVISAR» o
«Dígito: N. Confianza: P%. Decisión: ACEPTAR».
Copia N y P de leer_digito y la decisión de aplicar_regla. No respondas solo con
la decisión, aunque la última herramienta solo informe la política.
Después de esos tres campos puedes añadir el escalamiento indicado por la regla.
Al terminar una inspección con decisión, comprueba que incluiste dígito, porcentaje y decisión.
Si falta la imagen, pídela. Distingue una predicción de la etiqueta real.
Cada imagen adjunta es la del turno actual, incluso si hubo imágenes anteriores."""


@lru_cache(maxsize=1)
def recursos():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Configura GOOGLE_API_KEY en el archivo .env del proyecto.")
    modelo = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL") or MODELO_PREDETERMINADO,
        api_key=os.environ["GOOGLE_API_KEY"], temperature=0, max_tokens=1024,
        max_retries=1, timeout=45,
    )
    memoria = InMemorySaver()
    agente = create_agent(modelo, [leer_digito, aplicar_regla], context_schema=Contexto,
                          system_prompt=INSTRUCCIONES, checkpointer=memoria)
    return agente, memoria


def olvidar(hilo):
    if hilo and recursos.cache_info().currsize:
        recursos()[1].delete_thread(hilo)


def consultar(texto, imagen, hilo):
    agente, _ = recursos()
    adjunto = "Imagen adjunta." if imagen is not None else "Sin imagen adjunta."
    entrada = HumanMessage(content=f"{adjunto}\n{texto}", id=str(uuid4()))
    estado = agente.invoke(
        {"messages": [entrada]},
        config={"configurable": {"thread_id": hilo}, "recursion_limit": 10},
        context=Contexto(imagen),
    )
    mensajes = estado["messages"]
    inicio = next(i for i, m in enumerate(mensajes) if m.id == entrada.id)
    llamadas = [
        {"nombre": llamada["name"], "argumentos": llamada["args"]}
        for m in mensajes[inicio + 1:] if isinstance(m, AIMessage)
        for llamada in m.tool_calls
    ]
    return mensajes[-1].text, llamadas
