
import cv2 as cv
import numpy as np
import os
import time
from collections import deque

# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

CAMARA_ID = 0
ANCHO_PROCESO = 800

MAX_FEATURES = 750

MIN_MATCHES = 7
MIN_INLIERS = 5
MIN_SIMILITUD = 20

RATIO_TEST = 0.85

# Estabilidad de deteccion
HISTORIAL = deque(maxlen=4)
MIN_DETECCIONES = 2

# Seguimiento optico
MAX_PUNTOS = 80
MAX_FALLOS_TRACKING = 8

ARCHIVOS = [
    f"imgen_plantilla_{i}.jpg"
    for i in range(1, 10)
]

# =====================================================
# 2. OPTIMIZACION DE OPENCV
# =====================================================

cv.setUseOptimized(True)
cv.setNumThreads(2)

# =====================================================
# 3. CONFIGURAR ORB
# =====================================================

orb = cv.ORB_create(
    nfeatures=MAX_FEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=12,
    WTA_K=2,
    scoreType=cv.ORB_FAST_SCORE,
    patchSize=31,
    fastThreshold=10
)

matcher = cv.BFMatcher(
    cv.NORM_HAMMING,
    crossCheck=False
)

# =====================================================
# 4. CARGAR LAS NUEVE PLANTILLAS
# =====================================================

plantillas = []

print("\nCARGANDO PLANTILLAS...\n")

for archivo in ARCHIVOS:

    if not os.path.exists(archivo):
        print("No encontrada:", archivo)
        continue

    imagen = cv.imread(
        archivo,
        cv.IMREAD_GRAYSCALE
    )

    if imagen is None:
        continue

    h, w = imagen.shape

    escala = min(
        400 / max(h, w),
        1.0
    )

    imagen = cv.resize(
        imagen,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv.INTER_AREA
    )

    # Extraer puntos y descriptores
    puntos, descriptores = orb.detectAndCompute(
        imagen,
        None
    )

    if descriptores is None:
        print("Sin caracteristicas:", archivo)
        continue

    plantillas.append({
        "nombre": archivo,
        "imagen": imagen,
        "puntos": puntos,
        "descriptores": descriptores
    })

    print(
        archivo,
        "| Puntos:",
        len(puntos)
    )

if not plantillas:
    print("ERROR: No existen plantillas validas.")
    exit()

print("\nPlantillas cargadas:", len(plantillas))

# =====================================================
# 5. INICIAR CAMARA USB
# =====================================================

camara = cv.VideoCapture(
    CAMARA_ID,
    cv.CAP_DSHOW
)

if not camara.isOpened():

    camara.release()
    camara = cv.VideoCapture(CAMARA_ID)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la camara.")
    exit()

camara.set(
    cv.CAP_PROP_FOURCC,
    cv.VideoWriter_fourcc(*"MJPG")
)

camara.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
camara.set(cv.CAP_PROP_FPS, 30)
camara.set(cv.CAP_PROP_BUFFERSIZE, 1)

# =====================================================
# 6. CREAR LAS TRES VENTANAS
# =====================================================

cv.namedWindow(
    "DETECTOR ORB",
    cv.WINDOW_NORMAL
)

cv.namedWindow(
    "PUNTOS DE LA CAMARA",
    cv.WINDOW_NORMAL
)

cv.namedWindow(
    "PUNTOS DE LA PLANTILLA",
    cv.WINDOW_NORMAL
)

# =====================================================
# 7. VARIABLES DE SEGUIMIENTO
# =====================================================

poligono = None
puntos_tracking = None
frame_anterior = None

fallos_tracking = 0

ultima_similitud = 0
ultima_plantilla = ""

plantilla_actual = plantillas[0]

fps = 0
tiempo_anterior = time.perf_counter()

# =====================================================
# 8. FUNCION PARA INICIAR TRACKING
# =====================================================

def iniciar_tracking(gris, poligono):

    mascara = np.zeros(
        gris.shape,
        dtype=np.uint8
    )

    cv.fillConvexPoly(
        mascara,
        poligono.reshape(-1, 2),
        255
    )

    puntos = cv.goodFeaturesToTrack(
        gris,
        maxCorners=MAX_PUNTOS,
        qualityLevel=0.01,
        minDistance=5,
        mask=mascara,
        blockSize=7
    )

    return puntos

# =====================================================
# 9. FUNCION DE SEGUIMIENTO OPTICO
# =====================================================

def seguimiento_optico(
    anterior,
    actual,
    puntos,
    poligono
):

    if puntos is None or len(puntos) < 6:
        return None, None

    nuevos, estado, error = cv.calcOpticalFlowPyrLK(
        anterior,
        actual,
        puntos,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv.TERM_CRITERIA_EPS |
            cv.TERM_CRITERIA_COUNT,
            20,
            0.03
        )
    )

    if nuevos is None:
        return None, None

    validos = estado.reshape(-1) == 1

    origen = puntos[validos]
    destino = nuevos[validos]

    if len(origen) < 6:
        return None, None

    transformacion, mascara = cv.estimateAffinePartial2D(
        origen,
        destino,
        method=cv.RANSAC,
        ransacReprojThreshold=3.0
    )

    if transformacion is None or mascara is None:
        return None, None

    inliers = mascara.reshape(-1).astype(bool)

    if np.sum(inliers) < 6:
        return None, None

    nuevo_poligono = cv.transform(
        poligono.astype(np.float32),
        transformacion
    )

    nuevo_poligono = np.int32(nuevo_poligono)

    area = abs(cv.contourArea(nuevo_poligono))

    if area < 100 or area > actual.shape[0] * actual.shape[1] * 0.85:
        return None, None

    if not cv.isContourConvex(nuevo_poligono):
        return None, None

    nuevos_puntos = destino[inliers].reshape(-1, 1, 2)

    return nuevo_poligono, nuevos_puntos

# =====================================================
# 10. BUCLE PRINCIPAL
# =====================================================

print("\nDETECTOR INICIADO\n")

while True:

    ret, frame = camara.read()

    if not ret:
        break

    # Reducir resolucion conservando proporcion
    h, w = frame.shape[:2]

    escala = ANCHO_PROCESO / w
    alto = int(h * escala)

    frame = cv.resize(
        frame,
        (ANCHO_PROCESO, alto),
        interpolation=cv.INTER_AREA
    )

    gris = cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
    )

    # =================================================
    # 11. EXTRAER PUNTOS DE LA CAMARA
    # =================================================

    puntos_frame, descriptores_frame = orb.detectAndCompute(
        gris,
        None
    )

    # Crear imagen visual de los puntos de camara
    imagen_puntos_camara = cv.drawKeypoints(
        frame,
        puntos_frame,
        None,
        color=(0, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # =================================================
    # 12. COMPARAR LAS NUEVE PLANTILLAS
    # =================================================

    mejor_poligono = None
    mejor_inliers = 0
    mejor_similitud = 0
    mejores_matches = 0
    mejor_plantilla = None

    candidato_visual = None
    mejores_coincidencias = []

    if descriptores_frame is not None:

        candidatos = []

        for plantilla in plantillas:

            pares = matcher.knnMatch(
                plantilla["descriptores"],
                descriptores_frame,
                k=2
            )

            buenas = []

            for par in pares:

                if len(par) == 2:

                    m, n = par

                    if m.distance < RATIO_TEST * n.distance:
                        buenas.append(m)

            if len(buenas) > 0:

                candidatos.append(
                    (len(buenas), plantilla, buenas)
                )

        candidatos.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # Guardar la mejor candidata para visualizarla
        if candidatos:

            cantidad, candidato_visual, mejores_coincidencias = candidatos[0]

            plantilla_actual = candidato_visual

        # =================================================
        # 13. VALIDAR CANDIDATOS CON HOMOGRAFIA
        # =================================================

        for cantidad, plantilla, buenas in candidatos[:3]:

            if len(buenas) < MIN_MATCHES:
                continue

            src = np.float32([
                plantilla["puntos"][m.queryIdx].pt
                for m in buenas
            ]).reshape(-1, 1, 2)

            dst = np.float32([
                puntos_frame[m.trainIdx].pt
                for m in buenas
            ]).reshape(-1, 1, 2)

            H, mascara = cv.findHomography(
                src,
                dst,
                cv.RANSAC,
                4.0
            )

            if H is None or mascara is None:
                continue

            inliers = int(np.sum(mascara))

            similitud = (
                inliers / len(buenas)
            ) * 100

            if (
                inliers < MIN_INLIERS
                or similitud < MIN_SIMILITUD
            ):
                continue

            th, tw = plantilla["imagen"].shape

            esquinas = np.float32([
                [0, 0],
                [tw, 0],
                [tw, th],
                [0, th]
            ]).reshape(-1, 1, 2)

            transformadas = cv.perspectiveTransform(
                esquinas,
                H
            )

            candidato_poligono = np.int32(transformadas)

            area = abs(
                cv.contourArea(candidato_poligono)
            )

            if (
                area < 100
                or area > ANCHO_PROCESO * alto * 0.85
                or not cv.isContourConvex(candidato_poligono)
            ):
                continue

            if inliers > mejor_inliers:

                mejor_inliers = inliers
                mejor_similitud = similitud
                mejor_poligono = candidato_poligono

                mejor_plantilla = plantilla
                mejores_matches = len(buenas)

    # =================================================
    # 14. ACTUALIZAR TRACKING
    # =================================================

    deteccion_orb = mejor_poligono is not None

    if deteccion_orb:

        poligono = mejor_poligono

        ultima_similitud = mejor_similitud
        ultima_plantilla = mejor_plantilla["nombre"]

        puntos_tracking = iniciar_tracking(
            gris,
            poligono
        )

        fallos_tracking = 0

    else:

        if (
            frame_anterior is not None
            and poligono is not None
            and puntos_tracking is not None
            and fallos_tracking < MAX_FALLOS_TRACKING
        ):

            nuevo_poligono, nuevos_puntos = seguimiento_optico(
                frame_anterior,
                gris,
                puntos_tracking,
                poligono
            )

            if nuevo_poligono is not None:

                poligono = nuevo_poligono
                puntos_tracking = nuevos_puntos

                fallos_tracking += 1

            else:

                fallos_tracking = MAX_FALLOS_TRACKING

        else:

            fallos_tracking = MAX_FALLOS_TRACKING

    deteccion_tracking = (
        not deteccion_orb
        and poligono is not None
        and puntos_tracking is not None
        and fallos_tracking < MAX_FALLOS_TRACKING
    )

    HISTORIAL.append(
        deteccion_orb or deteccion_tracking
    )

    objeto_encontrado = (
        sum(HISTORIAL) >= MIN_DETECCIONES
        and poligono is not None
    )

    # =================================================
    # 15. MOSTRAR PUNTOS DE LA PLANTILLA
    # =================================================

    # Mostrar la plantilla que tiene mas coincidencias
    # Si no hay coincidencias, se conserva la anterior

    imagen_plantilla = plantilla_actual["imagen"]

    puntos_plantilla = plantilla_actual["puntos"]

    imagen_puntos_plantilla = cv.drawKeypoints(
        imagen_plantilla,
        puntos_plantilla,
        None,
        color=(255, 0, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv.putText(
        imagen_puntos_plantilla,
        plantilla_actual["nombre"],
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    # =================================================
    # 16. MOSTRAR ESTADO PRINCIPAL
    # =================================================

    if objeto_encontrado:

        estado = "Objeto encontrado"
        color = (0, 255, 0)

        cv.polylines(
            frame,
            [poligono],
            True,
            color,
            3
        )

        tipo = "ORB" if deteccion_orb else "TRACKING"

        cv.putText(
            frame,
            "Metodo: " + tipo,
            (10, 185),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        similitud_mostrar = ultima_similitud

    else:

        estado = "Buscando objeto"
        color = (0, 165, 255)

        similitud_mostrar = mejor_similitud

    # =================================================
    # 17. LABEL DE ESTADO
    # =================================================

    cv.rectangle(
        frame,
        (10, 10),
        (390, 50),
        (25, 25, 25),
        -1
    )

    cv.putText(
        frame,
        estado,
        (20, 38),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    # =================================================
    # 18. LABEL DE SIMILITUD
    # =================================================

    cv.rectangle(
        frame,
        (10, 60),
        (390, 100),
        (25, 25, 25),
        -1
    )

    cv.putText(
        frame,
        f"Similitud: {similitud_mostrar:.1f}%",
        (20, 88),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # =================================================
    # 19. INFORMACION ADICIONAL
    # =================================================

    cv.putText(
        frame,
        f"Matches: {mejores_matches}",
        (10, 125),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1
    )

    cv.putText(
        frame,
        f"Inliers: {mejor_inliers}",
        (10, 145),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1
    )

    if objeto_encontrado:

        cv.putText(
            frame,
            ultima_plantilla,
            (10, 165),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

    # =================================================
    # 20. CALCULAR FPS
    # =================================================

    ahora = time.perf_counter()

    fps_actual = 1 / max(
        ahora - tiempo_anterior,
        0.001
    )

    fps = (
        fps_actual
        if fps == 0
        else 0.85 * fps + 0.15 * fps_actual
    )

    tiempo_anterior = ahora

    cv.putText(
        frame,
        f"FPS: {fps:.1f}",
        (ANCHO_PROCESO - 110, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # =================================================
    # 21. MOSTRAR LAS TRES VENTANAS
    # =================================================

    cv.imshow(
        "DETECTOR ORB",
        frame
    )

    cv.imshow(
        "PUNTOS DE LA CAMARA",
        imagen_puntos_camara
    )

    cv.imshow(
        "PUNTOS DE LA PLANTILLA",
        imagen_puntos_plantilla
    )

    # =================================================
    # 22. GUARDAR FOTOGRAMA ANTERIOR
    # =================================================

    frame_anterior = gris.copy()

    if cv.waitKey(1) & 0xFF == 27:
        break

# =====================================================
# 23. CERRAR PROGRAMA
# =====================================================

camara.release()
cv.destroyAllWindows()

print("\nDETECTOR FINALIZADO")
