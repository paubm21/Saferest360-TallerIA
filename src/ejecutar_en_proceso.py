"""Alternativa local cuando el entorno no permite arrancar un kernel Jupyter.

Ejecuta celdas Python reales mediante IPython y captura sus salidas. No usa
respuestas simuladas. Solo acepta los dos cuadernos propios del proyecto.
"""
import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nombre", choices=["entrenamiento_u4", "evaluacion_agente_u5"])
    parser.add_argument("--solo-auditoria", action="store_true")
    args = parser.parse_args()
    if args.solo_auditoria and args.nombre != "entrenamiento_u4":
        parser.error("La auditoría pertenece a U4.")
    raiz = Path(__file__).resolve().parents[1]
    archivo = raiz / "notebooks" / f"{args.nombre}.ipynb"
    os.chdir(raiz)
    sys.path.insert(0, str(raiz))
    os.environ.setdefault("KERAS_BACKEND", "jax")
    nb = nbformat.read(archivo, as_version=4)
    temporal_ipython = TemporaryDirectory(prefix="ipython-taller-", ignore_cleanup_errors=True)
    os.environ["IPYTHONDIR"] = temporal_ipython.name
    shell = InteractiveShell.instance()
    fallo = None
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        if args.solo_auditoria and "auditoria" not in celda.metadata.get("tags", []):
            continue
        print(f"Ejecutando celda {i} en proceso...", flush=True)
        contador = shell.execution_count
        with capture_output() as capturada:
            resultado = shell.run_cell(celda.source, store_history=True)
        celda.execution_count = contador
        celda.outputs = []
        for nombre, texto in [("stdout", capturada.stdout), ("stderr", capturada.stderr)]:
            if texto:
                celda.outputs.append(nbformat.v4.new_output("stream", name=nombre, text=texto))
                print(texto, end="", flush=True)
        for salida in capturada.outputs:
            celda.outputs.append(nbformat.v4.new_output("display_data", data=salida.data,
                                                        metadata=salida.metadata or {}))
            if "text/plain" in salida.data:
                print(salida.data["text/plain"], flush=True)
        celda.metadata["ejecucion"] = "IPython en proceso; salida real capturada"
        fallo = resultado.error_before_exec or resultado.error_in_exec
        if fallo:
            celda.outputs.append(nbformat.v4.new_output("error", ename=type(fallo).__name__,
                evalue=str(fallo), traceback=[f"{type(fallo).__name__}: {fallo}"]))
            break
    temporal = archivo.with_suffix(".tmp")
    nbformat.write(nb, temporal)
    os.replace(temporal, archivo)
    if fallo:
        raise RuntimeError(f"El cuaderno falló; se conservaron sus salidas: {fallo}") from fallo
    print("Cuaderno guardado con salidas de la ejecución real.", flush=True)


if __name__ == "__main__":
    main()
