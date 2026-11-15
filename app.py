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
        if archivo is not None:
            mostrar_foto(archivo)

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

    st.info("La clasificación estará disponible cuando se conecte el modelo entrenado.")


if __name__ == "__main__":
    main()
