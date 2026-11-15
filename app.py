from hashlib import sha256
from importlib.util import find_spec

from PIL import Image, ImageOps
import streamlit as st


def mostrar_foto(archivo):
    try:
        with Image.open(archivo, formats=("JPEG", "PNG")) as foto:
            vista = ImageOps.exif_transpose(foto)
            vista.thumbnail((1200, 900))
            st.image(vista.convert("RGB"), caption=archivo.name)
    except (OSError, Image.DecompressionBombError):
        st.error("No se pudo abrir la foto. Selecciona otro archivo JPG o PNG.")
        return False
    return True


@st.cache_resource
def cargar_clasificador():
    from src.vision import ClasificadorZona

    return ClasificadorZona()


def mostrar_prediccion(datos_foto):
    st.subheader("Predicción del modelo")
    if find_spec("src.vision") is None:
        st.info("La predicción estará disponible cuando se integre el modelo entrenado.")
        return

    huella = sha256(datos_foto).digest()
    ultima_prediccion = st.session_state.get("prediccion_directa")
    if ultima_prediccion is None or ultima_prediccion[0] != huella:
        try:
            from src.vision import leer_imagen_bgr

            with st.spinner("Analizando la foto…"):
                imagen = leer_imagen_bgr(datos_foto)
                if imagen is None:
                    st.error("El modelo no pudo leer la foto. Prueba con otro archivo.")
                    return
                resultado = cargar_clasificador().predecir(imagen)
        except FileNotFoundError:
            st.warning("Faltan los archivos del modelo entrenado.")
            return
        except ImportError:
            st.warning("Instala las dependencias de visión para activar la predicción.")
            return
        except (OSError, ValueError):
            st.error("No se pudo obtener la predicción. Revisa la foto y el modelo.")
            return

        ultima_prediccion = huella, resultado
        st.session_state["prediccion_directa"] = ultima_prediccion

    _, resultado = ultima_prediccion
    estado, confianza = st.columns(2)
    estado.metric("Estado", resultado["estado"].capitalize())
    confianza.metric("Confianza", f"{resultado['confianza']:.1%}")
    st.caption(f"Modelo: {resultado['modelo']}")


def main():
    st.set_page_config(page_title="SafeRest 360 · Vigía de zonas", layout="wide")
    st.title("SafeRest 360 · Vigía de zonas")

    foto, detalles = st.columns([3, 2])

    with foto:
        st.subheader("Foto de la zona")
        archivo = st.file_uploader(
            "Sube una imagen JPG o PNG",
            type=["jpg", "jpeg", "png"],
            help="Usa una foto en la que se vea la zona seleccionada.",
        )
        foto_valida = archivo is not None and mostrar_foto(archivo)

    with detalles:
        st.subheader("Datos de la inspección")
        zona = st.selectbox(
            "Zona", ("bodega", "caja", "cocina"), format_func=str.capitalize, key="zona"
        )
        st.time_input("Hora de la foto", value=None, step=60, key="hora")
        if zona in ("caja", "cocina"):
            st.number_input(
                "Minutos que lleva sola la zona",
                min_value=0,
                value=None,
                step=1,
                key=f"minutos_{zona}",
                help="Completa este dato si sabes cuánto tiempo lleva vacía.",
            )

    if foto_valida:
        mostrar_prediccion(archivo.getvalue())


if __name__ == "__main__":
    main()
