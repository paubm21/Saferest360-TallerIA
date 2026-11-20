"""Ejecuta uno de los dos cuadernos con el kernel del entorno del proyecto."""
import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelspec import write_kernel_spec
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class KernelDelTaller(IPKernelApp):
    def load_config_file(self, *args, **kwargs):
        """Usa los valores de ipykernel y los argumentos del kernel privado."""

    def init_code(self):
        """Las celdas del cuaderno contienen todo el código de inicio del taller."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nombre", choices=["entrenamiento_u4", "evaluacion_agente_u5"])
    parser.add_argument("--solo-auditoria", action="store_true",
                        help="En U4 ejecuta solo la auditoría de los pesos, conservando el entrenamiento archivado.")
    args = parser.parse_args()
    nombre = args.nombre
    raiz = Path(__file__).resolve().parents[1]
    archivo = raiz / "notebooks" / f"{nombre}.ipynb"
    nb = nbformat.read(archivo, as_version=4)
    indices_auditoria = []
    original = nb
    if args.solo_auditoria:
        if nombre != "entrenamiento_u4":
            parser.error("--solo-auditoria solo aplica a entrenamiento_u4")
        indices_auditoria = [i for i, c in enumerate(nb.cells)
                            if "auditoria" in c.metadata.get("tags", [])]
        if not indices_auditoria:
            raise RuntimeError("No se encontraron celdas de auditoría.")
        nb = nbformat.v4.new_notebook(cells=[nb.cells[i] for i in indices_auditoria])
    fallo = None
    # Colab y el entorno del taller pueden tener distintos intérpretes y perfiles IPython.
    with TemporaryDirectory(prefix="jupyter-taller-") as temporal_jupyter:
        directorio = Path(temporal_jupyter)
        comando = [sys.executable, "-I", str(Path(__file__).resolve()), "--kernel",
                   "-f", "{connection_file}"]
        write_kernel_spec(directorio / "taller-digitos", overrides={"argv": comando})
        entorno = dict(os.environ, IPYTHONDIR=str(directorio / "ipython"),
                       MPLBACKEND="module://matplotlib_inline.backend_inline")
        cliente = NotebookClient(nb, timeout=900, kernel_name="taller-digitos",
                                 resources={"metadata": {"path": str(raiz)}})
        gestor = cliente.create_kernel_manager()
        gestor.kernel_spec_manager.kernel_dirs = [str(directorio)]
        print(f"Ejecutando {nombre} con {sys.executable}", flush=True)
        try:
            cliente.execute(env=entorno)
        except CellExecutionError as exc:
            fallo = exc
    if args.solo_auditoria:
        for i, celda in zip(indices_auditoria, nb.cells):
            original.cells[i] = celda
        nb = original
    temporal = archivo.with_suffix(".tmp")
    nbformat.write(nb, temporal)
    os.replace(temporal, archivo)
    for celda in nb.cells:
        for salida in celda.get("outputs", []):
            if salida.output_type == "stream":
                print(salida.text, end="")
            elif "text/plain" in salida.get("data", {}):
                print(salida.data["text/plain"])
    if fallo is not None:
        raise fallo


if __name__ == "__main__":
    if sys.argv[1:2] == ["--kernel"]:
        KernelDelTaller.launch_instance(argv=sys.argv[2:])
    else:
        main()
