
import cv2
import yt_dlp

# ==========================================
# CONFIGURACION
# ==========================================

URL = "https://www.youtube.com/watch?v=2juLrCH5w9U"

opciones = {
    "format": "230",
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "live_from_start": False,
    "remote_components": ["ejs:github"]
}

# ==========================================
# OBTENER URL DEL STREAMING
# ==========================================

print("Obteniendo streaming...")

try:

    with yt_dlp.YoutubeDL(opciones) as ydl:

        informacion = ydl.extract_info(
            URL,
            download=False
        )

        url_video = informacion["url"]

    print("Streaming obtenido correctamente")

except Exception as error:

    print("Error:", error)
    raise SystemExit


# ==========================================
# CONECTAR CON EL VIDEO
# ==========================================

configuracion = (
    "fflags;nobuffer|"
    "flags;low_delay|"
    "analyzeduration;1000000|"
    "probesize;1000000"
)

camara = cv2.VideoCapture(
    url_video,
    cv2.CAP_FFMPEG
)

camara.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not camara.isOpened():

    print("No se pudo abrir el streaming")
    raise SystemExit

print("Visualizando streaming...")


# ==========================================
# MOSTRAR FOTOGRAMAS
# ==========================================

while True:

    exito, fotograma = camara.read()
# ACÁ YA SE OBTUVO EL FOTOGRAMA ------------------------------------------
    if not exito:

        print("No se pudo recibir el fotograma")
        break

    # Mostrar fotograma original
    cv2.imshow(
        "Streaming de YouTube",
        fotograma
    )

    # Salir presionando Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# LIBERAR RECURSOS
# ==========================================

camara.release()
cv2.destroyAllWindows()

print("Streaming finalizado")
