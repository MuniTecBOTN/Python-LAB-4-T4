
import cv2
import numpy as np
import os
import time

# ==================================================
# CONFIGURACION
# ==================================================

RUTA_PLANTILLAS = "plantillas"

CAMARA_ID = 1

ANCHO_CAMARA = 640
ALTO_CAMARA = 480

AREA_MINIMA = 1500
AREA_MAXIMA = 100000

ANCHO_MINIMO = 30
ALTO_MINIMO = 30

NUM_CARACTERISTICAS = 500

MIN_COINCIDENCIAS = 8
MIN_INLIERS = 6

UMBRAL_RANSAC = 5.0
UMBRAL_INLIERS = 0.35

MAX_CANDIDATOS_RANSAC = 3

MOSTRAR_PLANTILLAS = True

# ==================================================
# CONFIGURAR ORB
# ==================================================

cv2.setUseOptimized(True)

orb = cv2.ORB_create(
    nfeatures=NUM_CARACTERISTICAS,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    fastThreshold=15
)

comparador = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

plantillas = {}


# ==================================================
# CARGAR PLANTILLAS
# ==================================================

def cargar_plantillas():

    if not os.path.exists(RUTA_PLANTILLAS):
        print("ERROR: No existe la carpeta de plantillas.")
        return False

    for nombre_pieza in sorted(os.listdir(RUTA_PLANTILLAS)):

        ruta_pieza = os.path.join(
            RUTA_PLANTILLAS,
            nombre_pieza
        )

        if not os.path.isdir(ruta_pieza):
            continue

        plantillas[nombre_pieza] = []

        for archivo in sorted(os.listdir(ruta_pieza)):

            ruta_imagen = os.path.join(
                ruta_pieza,
                archivo
            )

            imagen = cv2.imread(
                ruta_imagen,
                cv2.IMREAD_GRAYSCALE
            )

            if imagen is None:
                continue

            puntos, descriptores = orb.detectAndCompute(
                imagen,
                None
            )

            if descriptores is None:
                continue

            plantilla = {
                "nombre": nombre_pieza,
                "archivo": archivo,
                "imagen": imagen,
                "puntos": puntos,
                "descriptores": descriptores
            }

            plantillas[nombre_pieza].append(plantilla)

            print(
                f"[CARGADA] {nombre_pieza}/{archivo} "
                f"| Puntos: {len(puntos)}"
            )

            # Visualizar puntos ORB
            if MOSTRAR_PLANTILLAS:

                imagen_puntos = cv2.drawKeypoints(
                    imagen,
                    puntos,
                    None,
                    color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )

                nombre_ventana = f"{nombre_pieza} - {archivo}"

                cv2.namedWindow(
                    nombre_ventana,
                    cv2.WINDOW_NORMAL
                )

                cv2.imshow(
                    nombre_ventana,
                    imagen_puntos
                )

    if len(plantillas) != 3:
        print("ERROR: Se necesitan tres tipos de piezas.")
        return False

    for nombre, referencias in plantillas.items():

        if len(referencias) != 3:
            print(f"ERROR: {nombre} debe tener tres plantillas.")
            return False

    return True


# ==================================================
# RECONOCIMIENTO ORB
# ==================================================

def reconocer_pieza(puntos, descriptores):

    if descriptores is None or len(puntos) < 2:
        return "DESCONOCIDA", 0, 0

    candidatos = []

    # ----------------------------------------------
    # COMPARACION DE DESCRIPTORES
    # ----------------------------------------------

    for nombre_pieza, referencias in plantillas.items():

        for referencia in referencias:

            coincidencias = comparador.knnMatch(
                descriptores,
                referencia["descriptores"],
                k=2
            )

            buenas = []

            for coincidencia in coincidencias:

                if len(coincidencia) < 2:
                    continue

                primero, segundo = coincidencia

                if primero.distance < 0.75 * segundo.distance:
                    buenas.append(primero)

            if len(buenas) >= MIN_COINCIDENCIAS:

                candidatos.append({
                    "nombre": nombre_pieza,
                    "referencia": referencia,
                    "buenas": buenas
                })

    if not candidatos:
        return "DESCONOCIDA", 0, 0

    # Ordenar por cantidad de coincidencias
    candidatos.sort(
        key=lambda c: len(c["buenas"]),
        reverse=True
    )

    # Validar solamente los mejores candidatos
    candidatos = candidatos[:MAX_CANDIDATOS_RANSAC]

    mejor_pieza = "DESCONOCIDA"
    mejor_proporcion = 0
    mejor_inliers = 0

    # ----------------------------------------------
    # VALIDACION RANSAC
    # ----------------------------------------------

    for candidato in candidatos:

        referencia = candidato["referencia"]
        buenas = candidato["buenas"]

        puntos_origen = np.float32([
            puntos[m.queryIdx].pt
            for m in buenas
        ]).reshape(-1, 1, 2)

        puntos_destino = np.float32([
            referencia["puntos"][m.trainIdx].pt
            for m in buenas
        ]).reshape(-1, 1, 2)

        homografia, mascara = cv2.findHomography(
            puntos_destino,
            puntos_origen,
            cv2.RANSAC,
            UMBRAL_RANSAC
        )

        if homografia is None or mascara is None:
            continue

        inliers = int(np.sum(mascara))

        proporcion = inliers / len(buenas)

        if inliers < MIN_INLIERS:
            continue

        if proporcion < UMBRAL_INLIERS:
            continue

        if inliers > mejor_inliers:

            mejor_pieza = candidato["nombre"]
            mejor_inliers = inliers
            mejor_proporcion = proporcion

    return mejor_pieza, mejor_proporcion, mejor_inliers


# ==================================================
# DETECCION Y RECONOCIMIENTO
# ==================================================

def detectar_piezas(fotograma):

    gris = cv2.cvtColor(
        fotograma,
        cv2.COLOR_BGR2GRAY
    )

    suavizado = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    bordes = cv2.Canny(
        suavizado,
        50,
        150
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    bordes = cv2.morphologyEx(
        bordes,
        cv2.MORPH_CLOSE,
        kernel
    )

    contornos, _ = cv2.findContours(
        bordes,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cantidades = {
        nombre: 0
        for nombre in plantillas
    }

    desconocidas = 0

    # ==================================================
    # PROCESAR CADA CONTORNO
    # ==================================================

    for contorno in contornos:

        area = cv2.contourArea(contorno)

        if area < AREA_MINIMA or area > AREA_MAXIMA:
            continue

        x, y, ancho, alto = cv2.boundingRect(contorno)

        if ancho < ANCHO_MINIMO or alto < ALTO_MINIMO:
            continue

        # ----------------------------------------------
        # MASCARA SOBRE EL FOTOGRAMA COMPLETO
        # ----------------------------------------------

        mascara = np.zeros(
            gris.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            mascara,
            [contorno],
            -1,
            255,
            -1
        )

        # ----------------------------------------------
        # DETECTAR ORB DIRECTAMENTE EN EL FOTOGRAMA
        # ----------------------------------------------

        puntos, descriptores = orb.detectAndCompute(
            gris,
            mascara
        )

        # ----------------------------------------------
        # RECONOCER PIEZA
        # ----------------------------------------------

        nombre, puntuacion, inliers = reconocer_pieza(
            puntos,
            descriptores
        )

        # ----------------------------------------------
        # VISUALIZACION
        # ----------------------------------------------

        if nombre == "DESCONOCIDA":

            color = (0, 0, 255)
            desconocidas += 1

        else:

            color = (0, 255, 0)
            cantidades[nombre] += 1

        cv2.rectangle(
            fotograma,
            (x, y),
            (x + ancho, y + alto),
            color,
            2
        )

        texto = f"{nombre} | {inliers} inliers"

        cv2.putText(
            fotograma,
            texto,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return fotograma, cantidades, desconocidas


# ==================================================
# PROGRAMA PRINCIPAL
# ==================================================

def main():

    print("====================================")
    print(" SISTEMA ORB - BANDA TRANSPORTADORA")
    print("====================================")

    if not cargar_plantillas():

        cv2.destroyAllWindows()
        return

    camara = cv2.VideoCapture(CAMARA_ID)

    if not camara.isOpened():

        print("ERROR: No se pudo abrir la camara.")

        cv2.destroyAllWindows()
        return

    camara.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        ANCHO_CAMARA
    )

    camara.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        ALTO_CAMARA
    )

    nombre_ventana = "Reconocimiento ORB"

    cv2.namedWindow(
        nombre_ventana,
        cv2.WINDOW_NORMAL
    )

    tiempo_anterior = time.perf_counter()

    fps = 0

    print("\nCamara activada.")
    print("Presiona ESC para salir.")

    while True:

        exito, fotograma = camara.read()

        if not exito:
            break

        resultado, cantidades, desconocidas = detectar_piezas(
            fotograma
        )

        # Calcular FPS
        tiempo_actual = time.perf_counter()

        diferencia = tiempo_actual - tiempo_anterior

        tiempo_anterior = tiempo_actual

        if diferencia > 0:
            fps = 1 / diferencia

        # Panel de informacion
        cv2.rectangle(
            resultado,
            (0, 0),
            (260, 145),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            resultado,
            "RECONOCIMIENTO ORB",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        posicion_y = 52

        for nombre, cantidad in cantidades.items():

            cv2.putText(
                resultado,
                f"{nombre}: {cantidad}",
                (10, posicion_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

            posicion_y += 23

        cv2.putText(
            resultado,
            f"Desconocidas: {desconocidas}",
            (10, posicion_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

        cv2.putText(
            resultado,
            f"FPS: {fps:.1f}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            nombre_ventana,
            resultado
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camara.release()

    cv2.destroyAllWindows()

    print("Sistema finalizado.")


# ==================================================
# EJECUCION
# ==================================================

if __name__ == "__main__":

    main()
