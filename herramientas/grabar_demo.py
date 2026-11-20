"""Captura Gradio en localhost. En modo vivo cada respuesta procede de la app abierta."""
import argparse
import html
import json
import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import expect, sync_playwright


def grabar(raiz, destino, mostrar_notebook):
    caso = json.loads((raiz / "assets/caso_prueba.json").read_text(encoding="utf-8"))
    destino.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporal, sync_playwright() as pw:
        carpeta = Path(temporal)
        navegador = pw.chromium.launch(headless=True, executable_path=os.getenv("NAVEGADOR_TALLER"))
        pagina = navegador.new_page(viewport={"width": 1280, "height": 830}, device_scale_factor=1)
        pagina.set_default_timeout(100000)
        fotogramas = []

        def captura(titulo, segundos, nombre=None):
            # La banda inferior identifica el paso sin alterar los datos de la interfaz.
            pagina.evaluate("""titulo => {
                document.querySelector('#rotulo-demo')?.remove();
                const r = document.createElement('div');
                r.id = 'rotulo-demo';
                r.style.cssText = 'position:fixed;bottom:0;left:0;right:0;z-index:999999;padding:12px 22px;background:#14384c;color:white;font:600 19px system-ui;pointer-events:none';
                r.textContent = titulo;
                document.body.append(r);
            }""", titulo)
            foto = carpeta / f"paso-{len(fotogramas):02}.png"
            pagina.screenshot(path=foto, animations="disabled")
            fotogramas.append((foto, segundos))
            if nombre:
                (destino / nombre).write_bytes(foto.read_bytes())
            pagina.locator("#rotulo-demo").evaluate("e => e.remove()")

        pagina.goto("http://127.0.0.1:7860/", wait_until="domcontentloaded")
        cargar = pagina.get_by_role("button", name="Cargar el caso de prueba")
        cargar.wait_for()
        captura("Lector de casillas manuscritas   ·   Ruta 1   ·   Gradio", 4)
        pregunta = pagina.get_by_placeholder("Lee la casilla y decide si necesita revisión.")

        def consultar(texto, titulo, nombre):
            pregunta.fill(texto)
            captura(texto, 3)
            pagina.get_by_role("button", name="Consultar", exact=True).click()
            # La app vacía la pregunta junto con la respuesta y la traza al concluir el turno.
            expect(pregunta).to_have_value("", timeout=120000)
            captura(titulo, 7, nombre)

        if not mostrar_notebook:
            consultar("Hola, mi lote se llama Aliso. Salúdame.", "Saludo y nombre del lote", "demo-saludo.png")
        cargar.click()
        pagina.get_by_text(f"Dígito {caso['digito']}", exact=True).wait_for()
        expect(pagina.get_by_text(f"Confianza {caso['confianza_pct']:.2f} %", exact=False).first).to_be_visible()
        captura(f"Imagen {caso['indice']}   ·   Etiqueta del dataset: {caso['etiqueta_real']}   ·   Lectura directa de la CNN", 7, "demo-cnn.png")
        if mostrar_notebook:
            nb = json.loads((raiz / "notebooks/evaluacion_agente_u5.ipynb").read_text(encoding="utf-8"))
            salidas = nb["cells"][-1]["outputs"]
            registros = []
            resumen = []
            for salida in salidas:
                for linea in "".join(salida.get("text", "")).splitlines():
                    if " | llamadas reales:" in linea:
                        registros.append(linea)
                    if linea.startswith(("Casos con respuesta en vivo:", "Trayectorias correctas:", "Resultados correctos:")):
                        resumen.append(linea)
            if len(registros) != 4:
                raise ValueError("Se necesitan los cuatro registros guardados en la evaluación de Colab.")
            for indice, registro in enumerate(registros):
                nombre, datos = registro.split(" | llamadas reales:", 1)
                llamadas, respuesta = datos.split(" | respuesta:", 1)
                pagina.set_content("""<meta charset="utf-8"><style>
                    body{background:#f6f3ec;color:#153749;font:25px system-ui;margin:55px 64px}
                    header{font-size:20px;color:#486678}h1{font-size:48px;margin:20px 0}
                    h2{font-size:22px;margin:26px 0 6px}pre{font:24px Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
                    blockquote{font-size:32px;margin:10px 0;padding:15px 24px;border-left:6px solid #237b79;background:white}
                    </style><header>Unidad 5 · Extracto de la ejecución guardada en Colab</header>"""
                    + f"<h1>{html.escape(nombre)}</h1><h2>Llamadas registradas</h2><pre>{html.escape(llamadas.strip())}</pre>"
                    + f"<h2>Respuesta textual de Gemini</h2><blockquote>{html.escape(respuesta.strip())}</blockquote>"
                    + ("<p>" + "<br>".join(map(html.escape, resumen)) + "</p>" if indice == 3 else ""))
                captura("Fuente: notebooks/evaluacion_agente_u5.ipynb   ·   Salidas de la sesión entregada", 7)
        else:
            consultar("Lee el dígito e indica su confianza.", "Lectura de la CNN solicitada por el agente", "demo-lectura.png")
            consultar("Lee el dígito, indica su confianza y decide si aceptar o revisar la lectura.",
                      "Inspección de la casilla y herramientas del turno", "demo-inspeccion.png")
            consultar("¿Qué nombre te indiqué para mi lote?", "Memoria del hilo del saludo", "demo-memoria.png")
            pagina.get_by_role("button", name="Reiniciar conversación").click()
            expect(pagina.get_by_role("textbox", name="Herramientas del último turno")).to_have_value("Sin turnos")
            captura("Conversación reiniciada   ·   Historial vacío", 4, "demo-reinicio.png")
        navegador.close()
        # Concatena capturas de los estados reales y elimina únicamente la espera entre turnos.
        lista = carpeta / "fotogramas.txt"
        contenido = "".join(f"file '{p.name}'\nduration {duracion}\n" for p, duracion in fotogramas)
        lista.write_text(contenido + f"file '{fotogramas[-1][0].name}'\n", encoding="utf-8")
        video = destino / ("resultados-verificados.mp4" if mostrar_notebook else "demo-gradio.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "1", "-i", lista.name, "-vf", "fps=12",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(video.resolve())], cwd=carpeta, check=True)
        print(video.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proyecto", type=Path, required=True)
    parser.add_argument("--destino", type=Path, required=True)
    parser.add_argument("--mostrar-notebook", action="store_true",
                        help="Muestra los registros U5 guardados tras probar la lectura en la app.")
    args = parser.parse_args()
    grabar(args.proyecto, args.destino, args.mostrar_notebook)
