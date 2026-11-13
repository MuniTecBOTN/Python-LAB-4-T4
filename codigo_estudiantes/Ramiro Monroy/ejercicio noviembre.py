
import cv2
import numpy as np
import os
import time

# ============================================================
# SISTEMA DE INSPECCION DE PIEZAS A, B Y C
# SEGMENTACION DEL FONDO + ORB + RANSAC
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCHIVOS = {
    "PIEZA A": os.path.join(BASE_DIR, "pieza_A.jpg"),
    "PIEZA B": os.path.join(BASE_DIR, "pieza_B.jpg"),
    "PIEZA C": os.path.join(BASE_DIR, "pieza_C.jpg"),
}

# CAMARA
CAMARA_ID = 0
ANCHO_CAMARA = 640
ALTO_CAMARA = 480

# SEGMENTACION
SIGMA_FONDO = 25
UMBRAL_SEGMENTACION = 12

AREA_MINIMA = 250
AREA_MAXIMA_RELATIVA = 0.35

KERNEL_APERTURA = np.ones((3, 3), np.uint8)
KERNEL_CIERRE = np.ones((5, 5), np.uint8)

# ORB
NFEATURES = 5000
RATIO_TEST = 0.78

MINIMO_COINCIDENCIAS = 7
MINIMO_INLIERS = 5
MINIMA_PROPORCION_INLIERS = 0.25

UMBRAL_RANSAC = 5.0

orb = cv2.ORB_create(
    nfeatures=NFEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    fastThreshold=8
)

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 1. PROCESAMIENTO DE IMAGEN
# ============================================================

def procesar_gris(imagen):

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    gris = cv2.GaussianBlur(
        gris,
        (3, 3),
        0
    )

    return gris


# ============================================================
# 2. SEPARAR FIGURAS DEL FONDO BLANCO
# ============================================================

def crear_mascara_figuras(imagen):

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    # Estimar el fondo blanco
    fondo = cv2.GaussianBlur(
        gris,
        (0, 0),
        SIGMA_FONDO
    )

    # Comparar el fondo con la imagen original
    contraste = cv2.subtract(
        fondo,
        gris
    )

    # Separar las regiones oscuras
    _, mascara = cv2.threshold(
        contraste,
        UMBRAL_SEGMENTACION,
        255,
        cv2.THRESH_BINARY
    )

    # Eliminar ruido pequeño
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        KERNEL_APERTURA,
        iterations=1
    )

    # Cerrar pequeños huecos
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        KERNEL_CIERRE,
        iterations=2
    )

    return mascara


# ============================================================
# 3. DETECTAR CADA FIGURA INDIVIDUALMENTE
# ============================================================

def obtener_candidatos(imagen):

    mascara = crear_mascara_figuras(imagen)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    alto, ancho = imagen.shape[:2]

    area_maxima = (
        alto * ancho * AREA_MAXIMA_RELATIVA
    )

    candidatos = []

    for contorno in contornos:

        area = cv2.contourArea(contorno)

        if area < AREA_MINIMA:
            continue

        if area > area_maxima:
            continue

        x, y, w, h = cv2.boundingRect(contorno)

        if w < 8 or h < 8:
            continue

        # Agregar margen alrededor de la figura
        margen = 5

        x1 = max(0, x - margen)
        y1 = max(0, y - margen)

        x2 = min(ancho, x + w + margen)
        y2 = min(alto, y + h + margen)

        roi = imagen[y1:y2, x1:x2]

        mascara_roi = mascara[y1:y2, x1:x2]

        candidatos.append({
            "caja": (
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ),
            "roi": roi,
            "mascara": mascara_roi,
            "area": area
        })

    candidatos.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    return candidatos, mascara


# ============================================================
# 4. CARGAR Y RECORTAR LAS PLANTILLAS
# ============================================================

def extraer_plantilla(nombre, ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:

        raise FileNotFoundError(
            f"No se pudo cargar la plantilla: {ruta}"
        )

    mascara = crear_mascara_figuras(imagen)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_validos = [
        c for c in contornos
        if cv2.contourArea(c) >= AREA_MINIMA
    ]

    if not contornos_validos:

        raise ValueError(
            f"No se pudo separar la figura de {nombre}. "
            "Revisa la iluminacion y el contraste."
        )

    contorno = max(
        contornos_validos,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(contorno)

    margen = 5

    x1 = max(0, x - margen)
    y1 = max(0, y - margen)

    x2 = min(imagen.shape[1], x + w + margen)
    y2 = min(imagen.shape[0], y + h + margen)

    recorte = imagen[y1:y2, x1:x2]

    mascara_recorte = mascara[y1:y2, x1:x2]

    gris = procesar_gris(recorte)

    puntos, descriptores = orb.detectAndCompute(
        gris,
        mascara_recorte
    )

    if descriptores is None or len(puntos) < MINIMO_COINCIDENCIAS:

        cantidad = 0 if puntos is None else len(puntos)

        raise ValueError(
            f"{nombre} tiene pocos puntos ORB: {cantidad}"
        )

    print(
        f"{nombre}: {len(puntos)} puntos ORB "
        f"| Recorte: {recorte.shape[1]}x{recorte.shape[0]}"
    )

    return {
        "puntos": puntos,
        "descriptores": descriptores,
        "ancho": recorte.shape[1],
        "alto": recorte.shape[0]
    }


def cargar_plantillas():

    print("\n========== CARGANDO PLANTILLAS ==========\n")

    plantillas = {}

    for nombre, ruta in ARCHIVOS.items():

        plantillas[nombre] = extraer_plantilla(
            nombre,
            ruta
        )

    print("\nPlantillas cargadas correctamente.\n")

    return plantillas


# ============================================================
# 5. RECONOCER UNA FIGURA MEDIANTE ORB + RANSAC
# ============================================================

def comparar_roi_con_plantilla(candidato, plantilla):

    gris_roi = procesar_gris(
        candidato["roi"]
    )

    puntos_roi, descriptores_roi = orb.detectAndCompute(
        gris_roi,
        candidato["mascara"]
    )

    if descriptores_roi is None or len(puntos_roi) < 4:
        return None

    pares = matcher.knnMatch(
        plantilla["descriptores"],
        descriptores_roi,
        k=2
    )

    buenas = []

    for par in pares:

        if len(par) == 2:

            m, n = par

            if m.distance < RATIO_TEST * n.distance:

                buenas.append(m)

    if len(buenas) < MINIMO_COINCIDENCIAS:
        return None

    origen = np.float32([
        plantilla["puntos"][m.queryIdx].pt
        for m in buenas
    ]).reshape(-1, 1, 2)

    destino = np.float32([
        puntos_roi[m.trainIdx].pt
        for m in buenas
    ]).reshape(-1, 1, 2)

    homografia, mascara_inliers = cv2.findHomography(
        origen,
        destino,
        cv2.RANSAC,
        UMBRAL_RANSAC
    )

    if homografia is None or mascara_inliers is None:
        return None

    inliers = int(
        mascara_inliers.ravel().sum()
    )

    proporcion = inliers / max(len(buenas), 1)

    if inliers < MINIMO_INLIERS:
        return None

    if proporcion < MINIMA_PROPORCION_INLIERS:
        return None

    esquinas = np.float32([
        [0, 0],
        [plantilla["ancho"] - 1, 0],
        [plantilla["ancho"] - 1, plantilla["alto"] - 1],
        [0, plantilla["alto"] - 1]
    ]).reshape(-1, 1, 2)

    poligono = cv2.perspectiveTransform(
        esquinas,
        homografia
    )

    alto_roi, ancho_roi = candidato["roi"].shape[:2]

    area_proyectada = abs(
        cv2.contourArea(poligono)
    )

    area_roi = alto_roi * ancho_roi

    if area_proyectada < area_roi * 0.12:
        return None

    if area_proyectada > area_roi * 2.5:
        return None

    puntaje = inliers * proporcion

    return {
        "inliers": inliers,
        "coincidencias": len(buenas),
        "proporcion": proporcion,
        "puntaje": puntaje,
        "poligono": np.int32(poligono)
    }


# ============================================================
# 6. CLASIFICAR CADA FIGURA
# ============================================================

def clasificar_candidato(candidato, plantillas):

    mejor_nombre = None
    mejor_resultado = None

    for nombre, plantilla in plantillas.items():

        resultado = comparar_roi_con_plantilla(
            candidato,
            plantilla
        )

        if resultado is not None:

            if (
                mejor_resultado is None
                or resultado["puntaje"] > mejor_resultado["puntaje"]
            ):

                mejor_nombre = nombre
                mejor_resultado = resultado

    return mejor_nombre, mejor_resultado


# ============================================================
# 7. ACTIVAR LA CAMARA
# ============================================================

def abrir_camara():

    backends = [
        ("MSMF", cv2.CAP_MSMF),
        ("DSHOW", cv2.CAP_DSHOW)
    ]

    for nombre, backend in backends:

        print(f"Probando camara con {nombre}...")

        camara = cv2.VideoCapture(
            CAMARA_ID,
            backend
        )

        if camara.isOpened():

            correcto, frame = camara.read()

            if correcto and frame is not None and frame.size > 0:

                camara.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    ANCHO_CAMARA
                )

                camara.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    ALTO_CAMARA
                )

                print("Camara activada correctamente.\n")

                return camara

        camara.release()

    raise RuntimeError(
        "No se pudo abrir la camara."
    )


# ============================================================
# 8. SISTEMA PRINCIPAL
# ============================================================

def main():

    plantillas = cargar_plantillas()

    camara = abrir_camara()

    mostrar_mascara = True

    tiempo_anterior = time.time()

    print("SISTEMA DE INSPECCION INICIADO")
    print("ESC = Salir")
    print("M = Mostrar u ocultar mascara\n")

    try:

        while True:

            try:

                correcto, frame = camara.read()

            except cv2.error as error:

                print("Error al leer la camara:", error)

                break

            if not correcto or frame is None or frame.size == 0:

                print("No se recibio una imagen valida.")

                break

            # Separar figuras del fondo blanco
            candidatos, mascara = obtener_candidatos(frame)

            reconocidas = []

            # Analizar cada figura por separado
            for candidato in candidatos:

                nombre, resultado = clasificar_candidato(
                    candidato,
                    plantillas
                )

                x, y, w, h = candidato["caja"]

                if nombre is not None:

                    reconocidas.append(nombre)

                    color = (0, 255, 0)

                    etiqueta = (
                        f"{nombre} "
                        f"I:{resultado['inliers']} "
                        f"({resultado['proporcion']:.2f})"
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2
                    )

                    poligono = resultado["poligono"].copy()

                    poligono[:, 0, 0] += x
                    poligono[:, 0, 1] += y

                    cv2.polylines(
                        frame,
                        [poligono],
                        True,
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        etiqueta,
                        (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                else:

                    # Figura encontrada, pero sin clasificar
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 165, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "NO IDENTIFICADA",
                        (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 165, 255),
                        1
                    )

            # Mostrar resultados
            clases = set(reconocidas)

            alto, ancho = frame.shape[:2]

            cv2.rectangle(
                frame,
                (8, 8),
                (430, 105),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"REGIONES DETECTADAS: {len(candidatos)}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"PIEZAS RECONOCIDAS: {len(reconocidas)}",
                (18, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1
            )

            estado = "  ".join(
                f"{letra}: {'OK' if nombre in clases else '--'}"
                for letra, nombre in (
                    ("A", "PIEZA A"),
                    ("B", "PIEZA B"),
                    ("C", "PIEZA C")
                )
            )

            cv2.putText(
                frame,
                estado,
                (18, 83),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1
            )

            # Calcular FPS
            ahora = time.time()

            fps = 1.0 / max(
                ahora - tiempo_anterior,
                1e-6
            )

            tiempo_anterior = ahora

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (ancho - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "INSPECCION - SEGMENTACION + ORB",
                frame
            )

            if mostrar_mascara:

                cv2.imshow(
                    "MASCARA - FIGURAS EN BLANCO",
                    mascara
                )

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:

                break

            elif tecla in (ord("m"), ord("M")):

                mostrar_mascara = not mostrar_mascara

                if not mostrar_mascara:

                    cv2.destroyWindow(
                        "MASCARA - FIGURAS EN BLANCO"
                    )

    finally:

        camara.release()

        cv2.destroyAllWindows()

    print("\nSistema finalizado.")


# ============================================================
# 9. EJECUTAR PROGRAMA
# ============================================================

if __name__ == "__main__":

    main()
