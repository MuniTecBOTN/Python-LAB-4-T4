
import cv2
import numpy as np
import os
import time


# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

ANCHO_CAMARA = 960
ALTO_CAMARA = 540

ANCHO_ROI = 0.80
ALTO_ROI = 0.80

MINIMO_COINCIDENCIAS = 8
MINIMO_INLIERS = 6
MINIMA_PROPORCION_INLIERS = 0.35

RATIO_TEST = 0.80

INTERVALO_TRACKER = 15

# Reconocer el objeto cada 2 fotogramas
INTERVALO_ORB = 2

TAMANO_GAUSSIANO = (5, 5)
SIGMA_GAUSSIANO = 1.0

# Caracteristicas ORB utilizadas para reconocer
MAXIMO_CARACTERISTICAS = 500

# Puntos ORB que se dibujan en pantalla
MAXIMO_PUNTOS_VISIBLES = 70


# =====================================================
# 2. RUTA DE LA IMAGEN DE REFERENCIA
# =====================================================

carpeta_script = os.path.dirname(
    os.path.abspath(__file__)
)

ruta_imagen = os.path.join(
    carpeta_script,
    "Ramiro Monroy",
    "telefono.jpg"
)


# =====================================================
# 3. CARGAR Y PROCESAR LA PLANTILLA
# =====================================================

imagen_original = cv2.imread(ruta_imagen)

if imagen_original is None:
    raise FileNotFoundError(
        f"No se pudo cargar la imagen: {ruta_imagen}"
    )

# Convertir a escala de grises
plantilla_gris = cv2.cvtColor(
    imagen_original,
    cv2.COLOR_BGR2GRAY
)

# Aplicar filtro gaussiano
plantilla_gaussiana = cv2.GaussianBlur(
    plantilla_gris,
    TAMANO_GAUSSIANO,
    SIGMA_GAUSSIANO
)

# Mejorar contraste con CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

plantilla_mejorada = clahe.apply(
    plantilla_gaussiana
)

cv2.imshow(
    "Plantilla Gaussiano + CLAHE",
    plantilla_mejorada
)


# =====================================================
# 4. CONFIGURAR ORB
# =====================================================

orb = cv2.ORB_create(
    nfeatures=MAXIMO_CARACTERISTICAS,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31,
    fastThreshold=15
)

puntos_plantilla, descriptores_plantilla = (
    orb.detectAndCompute(
        plantilla_mejorada,
        None
    )
)

if descriptores_plantilla is None:
    raise RuntimeError(
        "No se encontraron caracteristicas en la plantilla."
    )

print(
    "Caracteristicas ORB de plantilla:",
    len(puntos_plantilla)
)


# =====================================================
# 5. SELECCIONAR PUNTOS ORB DISTRIBUIDOS
# =====================================================

def seleccionar_puntos_visuales(
    keypoints,
    ancho,
    alto,
    maximo=70,
    columnas=10,
    filas=8
):

    if not keypoints or ancho <= 0 or alto <= 0:
        return []

    # Crear una cuadricula para distribuir los puntos
    celdas = [
        [] for _ in range(columnas * filas)
    ]

    for punto in keypoints:

        x, y = punto.pt

        columna = min(
            int(x * columnas / ancho),
            columnas - 1
        )

        fila = min(
            int(y * filas / alto),
            filas - 1
        )

        indice = fila * columnas + columna

        celdas[indice].append(punto)

    # Ordenar los puntos por su respuesta ORB
    for celda in celdas:

        celda.sort(
            key=lambda p: p.response,
            reverse=True
        )

    seleccionados = []

    # Seleccionar puntos de diferentes regiones
    nivel = 0

    while len(seleccionados) < maximo:

        hubo_puntos = False

        for celda in celdas:

            if nivel < len(celda):

                seleccionados.append(
                    celda[nivel]
                )

                hubo_puntos = True

                if len(seleccionados) >= maximo:
                    break

        if not hubo_puntos:
            break

        nivel += 1

    return seleccionados


# Mostrar puntos ORB distribuidos en la plantilla

alto_plantilla, ancho_plantilla = plantilla_gris.shape

puntos_visuales_plantilla = seleccionar_puntos_visuales(
    puntos_plantilla,
    ancho_plantilla,
    alto_plantilla,
    MAXIMO_PUNTOS_VISIBLES
)

imagen_puntos = cv2.drawKeypoints(
    imagen_original,
    puntos_visuales_plantilla,
    None,
    color=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

cv2.imshow(
    "Caracteristicas ORB - Plantilla",
    imagen_puntos
)

print(
    "Puntos ORB mostrados en plantilla:",
    len(puntos_visuales_plantilla)
)


# =====================================================
# 6. CONFIGURAR BF MATCHER
# =====================================================

comparador = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# =====================================================
# 7. CREAR TRACKER CSRT
# =====================================================

def crear_tracker():

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if (
        hasattr(cv2, "legacy")
        and hasattr(cv2.legacy, "TrackerCSRT_create")
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT no disponible. Instala opencv-contrib-python."
    )


# =====================================================
# 8. VALIDAR CUADRILATERO
# =====================================================

def validar_cuadrilatero(puntos, ancho, alto):

    puntos = np.float32(puntos).reshape(4, 2)

    if not np.all(np.isfinite(puntos)):
        return False

    margen = 3

    if (
        np.any(puntos[:, 0] < margen)
        or np.any(puntos[:, 1] < margen)
        or np.any(puntos[:, 0] >= ancho - margen)
        or np.any(puntos[:, 1] >= alto - margen)
    ):
        return False

    contorno = puntos.reshape(-1, 1, 2)

    if not cv2.isContourConvex(contorno):
        return False

    area = abs(
        cv2.contourArea(contorno)
    )

    area_roi = ancho * alto

    if area < area_roi * 0.002:
        return False

    if area > area_roi * 0.85:
        return False

    return True


# =====================================================
# 9. RECONOCIMIENTO ORB + RANSAC
# =====================================================

def reconocer_objeto(frame):

    alto, ancho = frame.shape[:2]

    # Crear ROI central
    roi_ancho = int(ancho * ANCHO_ROI)
    roi_alto = int(alto * ALTO_ROI)

    x1 = (ancho - roi_ancho) // 2
    y1 = (alto - roi_alto) // 2

    x2 = x1 + roi_ancho
    y2 = y1 + roi_alto

    roi = frame[y1:y2, x1:x2]

    # Convertir ROI a escala de grises
    gris = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # Aplicar filtro gaussiano
    gris = cv2.GaussianBlur(
        gris,
        TAMANO_GAUSSIANO,
        SIGMA_GAUSSIANO
    )

    # Aplicar CLAHE
    gris = clahe.apply(gris)

    # Detectar puntos caracteristicos
    puntos_frame, descriptores_frame = (
        orb.detectAndCompute(
            gris,
            None
        )
    )

    if descriptores_frame is None:
        return None, puntos_frame, (x1, y1), 0, 0

    # Comparar descriptores
    coincidencias = comparador.knnMatch(
        descriptores_plantilla,
        descriptores_frame,
        k=2
    )

    # Aplicar Ratio Test
    buenas_coincidencias = []

    for par in coincidencias:

        if len(par) == 2:

            mejor, segundo = par

            if mejor.distance < RATIO_TEST * segundo.distance:
                buenas_coincidencias.append(mejor)

    total_coincidencias = len(
        buenas_coincidencias
    )

    if total_coincidencias < MINIMO_COINCIDENCIAS:
        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            0
        )

    # Coordenadas de la plantilla
    puntos_origen = np.float32([
        puntos_plantilla[m.queryIdx].pt
        for m in buenas_coincidencias
    ]).reshape(-1, 1, 2)

    # Coordenadas de la camara
    puntos_destino = np.float32([
        puntos_frame[m.trainIdx].pt
        for m in buenas_coincidencias
    ]).reshape(-1, 1, 2)

    # Calcular homografia con RANSAC
    try:

        homografia, mascara = cv2.findHomography(
            puntos_origen,
            puntos_destino,
            cv2.RANSAC,
            5.0
        )

    except cv2.error:

        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            0
        )

    if homografia is None or mascara is None:
        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            0
        )

    if not np.all(np.isfinite(homografia)):
        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            0
        )

    # Contar inliers
    inliers = int(mascara.sum())

    proporcion = inliers / total_coincidencias

    if inliers < MINIMO_INLIERS:
        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            inliers
        )

    if proporcion < MINIMA_PROPORCION_INLIERS:
        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            inliers
        )

    # Dimensiones de la plantilla
    alto_plantilla, ancho_plantilla = plantilla_gris.shape

    # Esquinas de la plantilla
    esquinas = np.float32([
        [0, 0],
        [ancho_plantilla - 1, 0],
        [ancho_plantilla - 1, alto_plantilla - 1],
        [0, alto_plantilla - 1]
    ]).reshape(-1, 1, 2)

    # Transformar esquinas
    esquinas_transformadas = cv2.perspectiveTransform(
        esquinas,
        homografia
    )

    puntos = esquinas_transformadas.reshape(4, 2)

    # Validar geometria dentro del ROI
    if not validar_cuadrilatero(
        puntos,
        roi.shape[1],
        roi.shape[0]
    ):
        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            inliers
        )

    # Convertir coordenadas ROI a coordenadas de camara
    esquinas_transformadas[:, 0, 0] += x1
    esquinas_transformadas[:, 0, 1] += y1

    puntos_globales = esquinas_transformadas.reshape(4, 2)

    # Calcular rectangulo envolvente
    bx = int(np.floor(np.min(puntos_globales[:, 0])))
    by = int(np.floor(np.min(puntos_globales[:, 1])))

    ex = int(np.ceil(np.max(puntos_globales[:, 0])))
    ey = int(np.ceil(np.max(puntos_globales[:, 1])))

    bx = max(0, bx)
    by = max(0, by)

    ex = min(ancho, ex)
    ey = min(alto, ey)

    w = ex - bx
    h = ey - by

    if w < 20 or h < 20:
        return (
            None,
            puntos_frame,
            (x1, y1),
            total_coincidencias,
            inliers
        )

    caja = (bx, by, w, h)

    informacion = {
        "caja": caja,
        "esquinas": puntos_globales,
        "coincidencias": total_coincidencias,
        "inliers": inliers,
        "proporcion": proporcion
    }

    return (
        informacion,
        puntos_frame,
        (x1, y1),
        total_coincidencias,
        inliers
    )


# =====================================================
# 10. ACTIVAR CAMARA
# =====================================================

camara = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camara.isOpened():
    raise RuntimeError(
        "No se pudo abrir la camara."
    )

camara.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    ANCHO_CAMARA
)

camara.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    ALTO_CAMARA
)

exito, primer_frame = camara.read()

if not exito or primer_frame is None:
    camara.release()

    raise RuntimeError(
        "No se pudo capturar el primer fotograma."
    )

print("Camara activada correctamente")

print(
    "Resolucion real:",
    primer_frame.shape[1],
    "x",
    primer_frame.shape[0]
)

cv2.imshow(
    "Prueba de camara",
    primer_frame
)

cv2.waitKey(500)
cv2.destroyWindow("Prueba de camara")


# =====================================================
# 11. VARIABLES DEL SISTEMA
# =====================================================

tracker = None
caja_tracker = None

contador_frames = 0
ultimo_tracker_frame = -INTERVALO_TRACKER

# Variables para controlar ORB
contador_orb = 0
ultimo_resultado_orb = None

# Puntos visuales almacenados para evitar
# seleccionarlos y ordenarlos en cada fotograma
puntos_visibles_cache = []

# Variables para calcular FPS suavizados
tiempo_anterior = time.perf_counter()
fps_suavizado = 0.0


# =====================================================
# 12. BUCLE PRINCIPAL
# =====================================================

print("Sistema ORB + GAUSSIANO + RANSAC + CSRT iniciado")
print("Coloca el objeto dentro del area de busqueda")
print("Presiona ESC para salir")


try:

    while True:

        exito, frame = camara.read()

        if not exito or frame is None:
            print("No se pudo obtener el fotograma")
            break

        contador_frames += 1

        alto, ancho = frame.shape[:2]

        # ---------------------------------------------
        # AREA ROI
        # ---------------------------------------------

        roi_ancho = int(ancho * ANCHO_ROI)
        roi_alto = int(alto * ALTO_ROI)

        x1 = (ancho - roi_ancho) // 2
        y1 = (alto - roi_alto) // 2

        x2 = x1 + roi_ancho
        y2 = y1 + roi_alto

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

        # ---------------------------------------------
        # DETECTAR OBJETO CADA 2 FOTOGRAMAS
        # ---------------------------------------------

        contador_orb += 1

        if (
            ultimo_resultado_orb is None
            or contador_orb >= INTERVALO_ORB
        ):

            ultimo_resultado_orb = reconocer_objeto(frame)

            contador_orb = 0

            # Actualizar los puntos visuales solamente
            # cuando se ejecuta nuevamente ORB

            (
                _,
                puntos_frame_nuevos,
                origen_roi_nuevo,
                _,
                _
            ) = ultimo_resultado_orb

            puntos_visibles_cache = []

            if puntos_frame_nuevos is not None:

                ox, oy = origen_roi_nuevo

                alto_roi = int(alto * ALTO_ROI)
                ancho_roi = int(ancho * ANCHO_ROI)

                puntos_visibles = seleccionar_puntos_visuales(
                    puntos_frame_nuevos,
                    ancho_roi,
                    alto_roi,
                    MAXIMO_PUNTOS_VISIBLES
                )

                puntos_visibles_cache = [
                    cv2.KeyPoint(
                        float(punto.pt[0] + ox),
                        float(punto.pt[1] + oy),
                        float(punto.size)
                    )
                    for punto in puntos_visibles
                ]

        # Recuperar el resultado ORB almacenado
        (
            informacion,
            puntos_frame,
            origen_roi,
            total_coincidencias,
            inliers
        ) = ultimo_resultado_orb

        # ---------------------------------------------
        # DIBUJAR PUNTOS ORB DISTRIBUIDOS
        # ---------------------------------------------

        cantidad_puntos = len(
            puntos_visibles_cache
        )

        if puntos_visibles_cache:

            frame = cv2.drawKeypoints(
                frame,
                puntos_visibles_cache,
                None,
                color=(0, 255, 0),
                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
            )

        # ---------------------------------------------
        # ACTUALIZAR O INICIAR CSRT
        # ---------------------------------------------

        if informacion is not None:

            caja_tracker = informacion["caja"]

            # Reiniciar tracker cada cierto intervalo
            if (
                tracker is None
                or contador_frames - ultimo_tracker_frame
                >= INTERVALO_TRACKER
            ):

                try:

                    x, y, w, h = caja_tracker

                    # Asegurar que la caja este dentro de la imagen
                    if (
                        x >= 0
                        and y >= 0
                        and w > 0
                        and h > 0
                        and x + w <= ancho
                        and y + h <= alto
                    ):

                        nuevo_tracker = crear_tracker()

                        nuevo_tracker.init(
                            frame,
                            (
                                int(x),
                                int(y),
                                int(w),
                                int(h)
                            )
                        )

                        tracker = nuevo_tracker
                        ultimo_tracker_frame = contador_frames

                except cv2.error as error:

                    tracker = None

                    print(
                        "Error al iniciar CSRT:",
                        error
                    )

        elif tracker is not None:

            # Si ORB no reconoce el objeto,
            # intentar mantener el seguimiento CSRT

            try:

                exito_tracker, nueva_caja = tracker.update(
                    frame
                )

                if exito_tracker:

                    caja_tracker = nueva_caja

                else:

                    tracker = None
                    caja_tracker = None

                    print(
                        "Seguimiento perdido. Buscando nuevamente..."
                    )

            except cv2.error as error:

                tracker = None
                caja_tracker = None

                print(
                    "Error durante seguimiento CSRT:",
                    error
                )

        # ---------------------------------------------
        # DIBUJAR CUADRO DE RECONOCIMIENTO
        # ---------------------------------------------

        if informacion is not None:

            esquinas = np.int32(
                informacion["esquinas"]
            )

            cv2.polylines(
                frame,
                [esquinas],
                True,
                (0, 255, 0),
                3
            )

            bx, by, bw, bh = informacion["caja"]

            cv2.putText(
                frame,
                "OBJETO RECONOCIDO",
                (bx, max(25, by - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        elif tracker is not None and caja_tracker is not None:

            x, y, w, h = [
                int(valor)
                for valor in caja_tracker
            ]

            x1_box = max(0, x)
            y1_box = max(0, y)

            x2_box = min(ancho, x + w)
            y2_box = min(alto, y + h)

            if x2_box > x1_box and y2_box > y1_box:

                cv2.rectangle(
                    frame,
                    (x1_box, y1_box),
                    (x2_box, y2_box),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "SEGUIMIENTO CSRT",
                    (x1_box, max(25, y1_box - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

        # ---------------------------------------------
        # PANEL DE INFORMACION
        # ---------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (390, 145),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            f"Puntos ORB visibles: {cantidad_puntos}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Coincidencias: {total_coincidencias}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Inliers RANSAC: {inliers}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if informacion is not None:

            estado = "RECONOCIDO"
            color_estado = (0, 255, 0)

        elif tracker is not None:

            estado = "EN SEGUIMIENTO"
            color_estado = (0, 255, 255)

        else:

            estado = "BUSCANDO OBJETO"
            color_estado = (0, 0, 255)

        cv2.putText(
            frame,
            estado,
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_estado,
            2
        )

        # ---------------------------------------------
        # CALCULAR FPS SUAVIZADOS
        # ---------------------------------------------

        tiempo_actual = time.perf_counter()

        diferencia = tiempo_actual - tiempo_anterior

        if diferencia > 0:

            fps_actual = 1.0 / diferencia

            if fps_suavizado == 0:

                fps_suavizado = fps_actual

            else:

                fps_suavizado = (
                    0.9 * fps_suavizado
                    + 0.1 * fps_actual
                )

        tiempo_anterior = tiempo_actual

        cv2.putText(
            frame,
            f"FPS: {fps_suavizado:.1f}",
            (20, alto - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ---------------------------------------------
        # MOSTRAR CAMARA
        # ---------------------------------------------

        cv2.imshow(
            "ORB - Gaussiano - RANSAC - CSRT",
            frame
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            print("Salida solicitada por el usuario")
            break


except KeyboardInterrupt:

    print("Programa interrumpido")


finally:

    camara.release()
    cv2.destroyAllWindows()

    print("Programa finalizado")
