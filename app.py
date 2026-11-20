"""Ejecuta python app.py desde la raíz del proyecto."""
import logging
from uuid import uuid4

import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

from src.agente import consultar, olvidar
from src.modelo import RAIZ, predecir


def nuevo_hilo():
    return str(uuid4())


def ejemplo():
    with Image.open(RAIZ / "assets" / "caso_prueba.png") as archivo:
        ampliada = archivo.convert("RGB").resize((192, 192), Image.Resampling.NEAREST)
        return np.asarray(ampliada)


def lectura(imagen):
    if imagen is None:
        return "Carga una imagen para ver su lectura."
    try:
        resultado = predecir(imagen)
    except ValueError as exc:
        return str(exc)
    return f"**Dígito {resultado['digito']}** · Confianza {resultado['confianza_pct']:.2f} %"


def responder(pregunta, imagen, historial, hilo):
    pregunta = pregunta.strip()
    if not pregunta:
        raise gr.Error("Escribe una pregunta para el agente.")
    try:
        respuesta, llamadas = consultar(pregunta, imagen, hilo)
    except RuntimeError as exc:
        if "Configura GOOGLE_API_KEY" in str(exc):
            raise gr.Error(str(exc)) from None
        raise gr.Error("El agente no pudo completar el turno. Reinicia la conversación y reintenta.") from None
    except Exception as exc:
        logging.warning("Fallo de consulta: %s", type(exc).__name__)
        raise gr.Error("Consulta interrumpida. Comprueba conexión, modelo y cuota de Gemini.") from None
    historial = [*(historial or []), {"role": "user", "content": pregunta},
                 {"role": "assistant", "content": respuesta}]
    traza = " → ".join(
        f"{c['nombre']}({', '.join(f'{k}={v}' for k, v in c['argumentos'].items())})"
        for c in llamadas
    ) or "Ninguna"
    return historial, traza, ""


def reiniciar(hilo):
    olvidar(hilo)
    return nuevo_hilo(), [], "Sin turnos", ""


def interfaz():
    with gr.Blocks(title="Lector de casillas") as demo:
        gr.Markdown("# Lector de casillas\nPrototipo educativo con dígitos de 8 × 8. "
                    "Su desempeño en formularios reales no ha sido validado.")
        hilo = gr.State(value=nuevo_hilo, time_to_live=3600, delete_callback=olvidar)
        with gr.Row():
            with gr.Column(scale=2):
                imagen = gr.Image(type="numpy", image_mode="RGB", format="png", sources=["upload"],
                                  label="Recorte de un solo dígito", height=230)
                cargar = gr.Button("Cargar el caso de prueba")
                gr.Markdown("### Lectura directa de la CNN")
                directa = gr.Markdown(lectura(None))
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Agente de inspección", height=265)
                pregunta = gr.Textbox(label="Pregunta", max_lines=3,
                                      placeholder="Lee la casilla y decide si necesita revisión.")
                with gr.Row():
                    enviar = gr.Button("Consultar", variant="primary")
                    reset = gr.Button("Reiniciar conversación")
                traza = gr.Textbox(value="Sin turnos", label="Herramientas del último turno",
                                   interactive=False)
        cargar.click(ejemplo, outputs=imagen)
        imagen.change(lectura, imagen, directa)
        gr.on([enviar.click, pregunta.submit], responder, [pregunta, imagen, chat, hilo],
              [chat, traza, pregunta], concurrency_limit=1)
        reset.click(reiniciar, hilo, [hilo, chat, traza, pregunta])
    return demo


if __name__ == "__main__":
    # La aplicación se sirve en la interfaz de bucle local del equipo.
    web = gr.mount_gradio_app(FastAPI(), interfaz(), path="/")
    uvicorn.run(web, host="127.0.0.1", port=7860, log_level="warning")
