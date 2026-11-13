
import cv2
import os

# ==========================================
# 1. CONFIGURACION
# ==========================================

RUTA_PLANTILLAS = "plantillas"

ARCHIVOS = {
    "PIEZA A": "pieza_A.jpg",
    "PIEZA B": "pieza_B.jpg",
    "PIEZA C": "pieza_C.jpg"
}

# Crear detector ORB
orb = cv2.ORB_create(nfeatures=1000)

# ==========================================
# 2. CARGAR PLANTILLAS
# ==========================================

plantillas = {}

for nombre, archivo in ARCHIVOS.items():

    ruta = os.path.join(RUTA_PLANTILLAS, archivo)

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo cargar: {ruta}"
        )

    # Convertir a escala de grises
    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    # Detectar puntos clave y descriptores
    puntos, descriptores = orb.detectAndCompute(
        gris, None
    )

    if descriptores is None:
        raise ValueError(
            f"No se encontraron características en {nombre}"
        )

    plantillas[nombre] = {
        "imagen": imagen,
        "puntos": puntos,
        "descriptores": descriptores
    }

    print("--------------------------------")
    print("Plantilla:", nombre)
    print("Puntos ORB:", len(puntos))
    print("Descriptores:", descriptores.shape)

print("--------------------------------")
print("LAS 3 PLANTILLAS SE CARGARON CORRECTAMENTE")
