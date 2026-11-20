"""Sincroniza solo el estado de U5 con el registro real de la última ejecución."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main():
    r = json.loads((RAIZ / "assets/evaluacion_actual.json").read_text(encoding="utf-8"))
    estado = r["estado"]
    completa = estado == "COMPLETA"
    aprobada = completa and r["casos_ejecutados"] == r["casos_esperados"] and all(
        f["trayectoria_ok"] and f["resultado_ok"] and
        f["confianza_transmitida_ok"] is not False for f in r["filas"])
    texto = (f"**Evaluación actual:** {estado}. Casos ejecutados con Gemini: "
             f"**{r['casos_ejecutados']}/{r['casos_esperados']}**; trayectorias correctas: "
             f"**{r['trayectorias_correctas']}**; resultados correctos: **{r['resultados_correctos']}**. "
             f"Modelo registrado: `{r['modelo']}`.\n\n")
    texto += ("Los cuatro casos cumplen los criterios de esta ejecución. " if aprobada else
              "La corrección todavía no cuenta con una evaluación completa aprobada en vivo. ")
    texto += ("Ver el [notebook actual](notebooks/evaluacion_agente_u5.ipynb) y "
              "el [registro de ejecución](assets/evaluacion_actual.json). "
              "Las pruebas locales del validador no son respuestas generadas por Gemini.")
    if r["filas"]:
        texto += "\n\n| Caso | Herramientas observadas | Respuesta real | Cumple |\n| --- | --- | --- | --- |\n"
        for f in r["filas"]:
            respuesta = f["respuesta"].replace("|", "\\|").replace("\n", " ")
            cumple = f["trayectoria_ok"] and f["resultado_ok"] and f["confianza_transmitida_ok"] is not False
            texto += f"| {f['caso']} | {' → '.join(f['invocadas']) or 'Ninguna'} | {respuesta} | {'Sí' if cumple else 'No'} |\n"
    p = RAIZ / "README.md"
    inicio, fin = "<!-- ESTADO_U5_INICIO -->", "<!-- ESTADO_U5_FIN -->"
    contenido = p.read_text(encoding="utf-8")
    antes, resto = contenido.split(inicio, 1)
    _, despues = resto.split(fin, 1)
    p.write_text(antes + inicio + "\n" + texto + "\n" + fin + despues, encoding="utf-8")
    print("Estado del README actualizado desde evaluacion_actual.json:", estado)


if __name__ == "__main__":
    main()
