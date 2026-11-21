import cv2
import numpy as np
import os
import time

# ============================================================
# 1. CONFIGURACION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCHIVOS = {
    "PIEZA A": "pieza_A.jpg",
    "PIEZA B": "pieza_B.jpg",
    "PIEZA C": "pieza_C.jpg"
}

ANCHO_CAMARA = 1280
ALTO_CAMARA = 720

# Deteccion de objetos mediante comparacion con el fondo
UMBRAL_DIFERENCIA = 22
AREA_MINIMA = 1200
MARGEN_RECORTE = 0.18

# Reconocimiento ORB
NUM_PUNTOS_ORB = 2500
RATIO_TEST = 0.78
MINIMO_COINCIDENCIAS = 6
MINIMO_INLIERS = 5
MINIMA_PROPORCION_INLIERS = 0.30

# Para evitar imprimir cientos de mensajes por segundo
INTERVALO_DIAGNOSTICO = 1.0

# ============================================================
# 2. PREPROCESAMIENTO Y ORB
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

orb = cv2.ORB_create(
    nfeatures=NUM_PUNTOS_ORB,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31,
    fastThreshold=7
)

comparador = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def preparar_gris(imagen):
    """Convierte a gris y mejora contraste de forma consistente."""
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    gris = clahe.apply(gris)
    return gris


# ============================================================
# 3. CARGAR PLANTILLAS
# ============================================================

plantillas = {}

print("\n========== CARGANDO PLANTILLAS ==========")

for nombre, archivo in ARCHIVOS.items():
    ruta = os.path.join(BASE_DIR, archivo)
    imagen = cv2.imread(ruta)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo cargar: {ruta}\n"
            "Comprueba que las tres imagenes esten en la misma carpeta "
            "que este archivo .py."
        )

    gris = preparar_gris(imagen)
    puntos, descriptores = orb.detectAndCompute(gris, None)

    if descriptores is None or len(puntos) < 4:
        raise ValueError(
            f"{nombre}: no se encontraron suficientes puntos ORB "
            f"en {archivo}. Prueba una fotografia mas nitida y con "
            "el dibujo completo."
        )

    plantillas[nombre] = {
        "imagen": gris,
        "puntos": puntos,
        "descriptores": descriptores
    }

    print(f"{nombre}: {len(puntos)} puntos ORB | archivo: {archivo}")

print("PLANTILLAS CARGADAS CORRECTAMENTE\n")


# ============================================================
# 4. RECONOCER UNA PIEZA
# ============================================================

def reconocer_pieza(imagen_candidata):
    """
    Compara el recorte completo contra A, B y C.
    IMPORTANTE: no se usa la mascara de movimiento para ORB,
    porque podria eliminar los dibujos negros y grises.
    """

    gris = preparar_gris(imagen_candidata)
    puntos_candidatos, descriptores_candidatos = orb.detectAndCompute(
        gris, None
    )

    diagnostico = {}
    if descriptores_candidatos is None or len(puntos_candidatos) < 4:
        return "NO EVALUABLE", 0, diagnostico

    mejor_nombre = "DESCONOCIDA"
    mejor_puntuacion = 0
    mejor_inliers = 0

    for nombre, plantilla in plantillas.items():

        pares = comparador.knnMatch(
            plantilla["descriptores"],
            descriptores_candidatos,
            k=2
        )

        buenas = []
        for par in pares:
            if len(par) == 2:
                m, n = par
                if m.distance < RATIO_TEST * n.distance:
                    buenas.append(m)

        if len(buenas) < MINIMO_COINCIDENCIAS:
            diagnostico[nombre] = (len(buenas), 0, 0.0)
            continue

        origen = np.float32([
            plantilla["puntos"][m.queryIdx].pt for m in buenas
        ]).reshape(-1, 1, 2)

        destino = np.float32([
            puntos_candidatos[m.trainIdx].pt for m in buenas
        ]).reshape(-1, 1, 2)

        try:
            homografia, mascara_ransac = cv2.findHomography(
                origen,
                destino,
                cv2.RANSAC,
                5.0
            )
        except cv2.error:
            homografia, mascara_ransac = None, None

        if homografia is None or mascara_ransac is None:
            diagnostico[nombre] = (len(buenas), 0, 0.0)
            continue

        inliers = int(mascara_ransac.ravel().sum())
        proporcion = inliers / max(len(buenas), 1)
        diagnostico[nombre] = (len(buenas), inliers, proporcion)

        # Se exige una cantidad minima y una proporcion razonable.
        if (
            inliers >= MINIMO_INLIERS
            and proporcion >= MINIMA_PROPORCION_INLIERS
        ):
            # Se priorizan los inliers y luego la proporcion.
            puntuacion = inliers + proporcion

            if puntuacion > mejor_puntuacion:
                mejor_puntuacion = puntuacion
                mejor_nombre = nombre
                mejor_inliers = inliers

    return mejor_nombre, mejor_inliers, diagnostico


# ============================================================
# 5. ABRIR CAMARA
# ============================================================

print("ACTIVANDO CAMARA...")

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camara.isOpened():
    camara.release()
    camara = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camara.isOpened():
    raise RuntimeError(
        "No se pudo activar la camara. Cierra otras aplicaciones "
        "que puedan estar utilizandola."
    )

camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)

# Reduce el buffer para intentar procesar imagenes recientes.
camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("CAMARA ACTIVADA")


# ============================================================
# 6. CAPTURAR EL FONDO VACIO
# ============================================================

print("\nDeja la banda/superficie completamente vacia.")
print("Capturando el fondo durante unos segundos...")

fotogramas_fondo = []

for i in range(30):
    correcto, frame = camara.read()

    if not correcto:
        camara.release()
        raise RuntimeError("No se pudo capturar el fondo.")

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fotogramas_fondo.append(gris)

    vista = frame.copy()
    cv2.putText(
        vista,
        f"CAPTURANDO FONDO {i + 1}/30",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )
    cv2.imshow("CALIBRACION", vista)
    cv2.waitKey(30)

fondo = np.median(
    np.array(fotogramas_fondo),
    axis=0
).astype(np.uint8)

fondo = cv2.GaussianBlur(fondo, (5, 5), 0)
cv2.destroyWindow("CALIBRACION")

print("FONDO CAPTURADO")
print("\nINSPECCION INICIADA. Presiona ESC para salir.")
print("Coloca las piezas dentro del encuadre de la camara.\n")


# ============================================================
# 7. INSPECCION EN TIEMPO REAL
# ============================================================

ultimo_diagnostico = 0.0
# ============================================================
# CONTADORES ACUMULATIVOS DE PIEZAS
# ============================================================

TIEMPO_ESTABILIDAD = 0.35
TIEMPO_REARME = 1.0

contadores = {
    "PIEZA A": 0,
    "PIEZA B": 0,
    "PIEZA C": 0
}

inicio_presencia = {
    "PIEZA A": None,
    "PIEZA B": None,
    "PIEZA C": None
}

ultima_deteccion = {
    "PIEZA A": 0.0,
    "PIEZA B": 0.0,
    "PIEZA C": 0.0
}

ya_contada = {
    "PIEZA A": False,
    "PIEZA B": False,
    "PIEZA C": False
}

try:
    while True:

        correcto, frame = camara.read()
        if not correcto:
            print("No se pudo leer un fotograma de la camara.")
            break

        alto_frame, ancho_frame = frame.shape[:2]

        gris_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris_frame = cv2.GaussianBlur(gris_frame, (5, 5), 0)

        # La mascara se usa SOLAMENTE para localizar objetos.
        diferencia = cv2.absdiff(fondo, gris_frame)

        _, mascara = cv2.threshold(
            diferencia,
            UMBRAL_DIFERENCIA,
            255,
            cv2.THRESH_BINARY
        )

        # Unir partes cercanas del dibujo para formar una region
        # candidata mas completa. Si las piezas estan muy juntas,
        # reduce el kernel a (5, 5).
        kernel_cierre = np.ones((9, 9), np.uint8)
        kernel_limpieza = np.ones((3, 3), np.uint8)

        mascara = cv2.morphologyEx(
            mascara,
            cv2.MORPH_CLOSE,
            kernel_cierre,
            iterations=2
        )

        mascara = cv2.morphologyEx(
            mascara,
            cv2.MORPH_OPEN,
            kernel_limpieza,
            iterations=1
        )

        contornos, _ = cv2.findContours(
            mascara.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Ordenar de izquierda a derecha para facilitar la lectura.
        contornos = sorted(
            contornos,
            key=lambda c: cv2.boundingRect(c)[0]
        )

        cantidad_detectada = 0
        resultados_diagnostico = []
        clases_presentes = set()

        for contorno in contornos:

            area = cv2.contourArea(contorno)
            if area < AREA_MINIMA:
                continue

            x, y, ancho, alto = cv2.boundingRect(contorno)

            if ancho < 10 or alto < 10:
                continue

            # Agregar margen alrededor del contorno. Esto ayuda si
            # el contorno se formo principalmente con el dibujo.
            margen_x = int(ancho * MARGEN_RECORTE)
            margen_y = int(alto * MARGEN_RECORTE)

            x1 = max(0, x - margen_x)
            y1 = max(0, y - margen_y)
            x2 = min(ancho_frame, x + ancho + margen_x)
            y2 = min(alto_frame, y + alto + margen_y)

            recorte = frame[y1:y2, x1:x2]

            if recorte.size == 0:
                continue

            # ORB analiza el recorte visual completo, NO la mascara.
            nombre, inliers, diagnostico = reconocer_pieza(recorte)

            cantidad_detectada += 1
            resultados_diagnostico.append((nombre, diagnostico))

            if nombre in contadores:
                clases_presentes.add(nombre)
                color = (0, 220, 0)       # Verde: reconocida
            elif nombre == "DESCONOCIDA":
                color = (0, 0, 255)       # Rojo: no coincide
            else:
                color = (0, 255, 255)     # Amarillo: no evaluable

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            texto = f"{nombre} | inliers: {inliers}"
            cv2.putText(
                frame,
                texto,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2
            )

        # ====================================================
        # CONTABILIZAR PIEZAS A, B Y C
        # ====================================================

        ahora = time.time()

        for nombre in contadores:

            if nombre in clases_presentes:

                if inicio_presencia[nombre] is None:
                    inicio_presencia[nombre] = ahora

                ultima_deteccion[nombre] = ahora

                if (
                    not ya_contada[nombre]
                    and ahora - inicio_presencia[nombre] >= TIEMPO_ESTABILIDAD
                ):
                    contadores[nombre] += 1
                    ya_contada[nombre] = True

                    print(
                        f"CONTABILIZADA: {nombre} | "
                        f"TOTAL: {contadores[nombre]}"
                    )

            else:

                if (
                    inicio_presencia[nombre] is not None
                    and ahora - ultima_deteccion[nombre] >= TIEMPO_REARME
                ):
                    inicio_presencia[nombre] = None
                    ya_contada[nombre] = False

                # ====================================================
        # MOSTRAR CONTADORES EN PANTALLA
        # ====================================================

        total_reconocidas = sum(contadores.values())

        cv2.rectangle(
            frame,
            (10, 48),
            (330, 195),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            f"PIEZA A: {contadores['PIEZA A']}",
            (22, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"PIEZA B: {contadores['PIEZA B']}",
            (22, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"PIEZA C: {contadores['PIEZA C']}",
            (22, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"TOTAL RECONOCIDAS: {total_reconocidas}",
            (22, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"OBJETOS DETECTADOS: {cantidad_detectada}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )

        cv2.imshow("INSPECCION A B C", frame)
        cv2.imshow("MASCARA DE DETECCION", mascara)

        # Imprimir diagnostico como maximo una vez por segundo.
        ahora = time.time()
        if ahora - ultimo_diagnostico >= INTERVALO_DIAGNOSTICO:
            if resultados_diagnostico:
                print("\n--- DIAGNOSTICO ORB ---")
                for indice, (nombre, datos) in enumerate(
                    resultados_diagnostico, start=1
                ):
                    print(f"Objeto {indice}: resultado = {nombre}")
                    for plantilla, valores in datos.items():
                        coincidencias, inliers, proporcion = valores
                        print(
                            f"  {plantilla}: "
                            f"coincidencias={coincidencias}, "
                            f"inliers={inliers}, "
                            f"proporcion={proporcion:.2f}"
                        )
            ultimo_diagnostico = ahora

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break

finally:
    camara.release()
    cv2.destroyAllWindows()
    print("\nSISTEMA FINALIZADO")
