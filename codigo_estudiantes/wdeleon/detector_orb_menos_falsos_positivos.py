import cv2 as cv
import numpy as np
import os
import time
from collections import deque

# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

CAMARA_ID = 0

# Reducir a 640 suele mejorar bastante el rendimiento.
# Si la deteccion pierde precision, probar con 720.
ANCHO_PROCESO = 640

# ORB se ejecuta cada cierto numero de frames mientras
# Lucas-Kanade mantenga correctamente el seguimiento.
MAX_FEATURES = 600
ORB_INTERVALO = 8

# Filtros mas estrictos para evitar reconocer objetos del fondo.
MIN_MATCHES = 12
MIN_INLIERS = 9
MIN_SIMILITUD = 45
RATIO_TEST = 0.78

# Geometria minima para aceptar una homografia.
MIN_AREA_OBJETO = 0.002       # 0.2% del area del fotograma
MAX_AREA_OBJETO = 0.70        # 70% del area del fotograma
MIN_EXTENSION_PLANTILLA = 0.12
MARGEN_FRAME = 8

# Seguimiento optico
MAX_PUNTOS = 70
MIN_PUNTOS_TRACKING = 9

# Visualizacion de puntos de camara:
# se actualiza cada 3 frames para reducir carga.
INTERVALO_PUNTOS_VISUALES = 3
MAX_PUNTOS_VISUALES = 200

HISTORIAL = deque(maxlen=4)
MIN_DETECCIONES = 2

ARCHIVOS = [
    f"imgen_plantilla_{i}.jpg"
    for i in range(1, 10)
]

# =====================================================
# 2. OPTIMIZACION DE OPENCV
# =====================================================

cv.setUseOptimized(True)
cv.setNumThreads(2)

# ORB principal: se usa para reconocer la pieza.
orb = cv.ORB_create(
    nfeatures=MAX_FEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=12,
    WTA_K=2,
    scoreType=cv.ORB_FAST_SCORE,
    patchSize=31,
    fastThreshold=15
)

# ORB ligero: solo dibuja puntos en la ventana visual.
orb_visual = cv.ORB_create(
    nfeatures=MAX_PUNTOS_VISUALES,
    scaleFactor=1.2,
    nlevels=4,
    edgeThreshold=12,
    scoreType=cv.ORB_FAST_SCORE,
    patchSize=31,
    fastThreshold=15
)

matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

# =====================================================
# 3. CARGAR LAS NUEVE PLANTILLAS
# =====================================================

plantillas = []

print("\nCARGANDO PLANTILLAS...\n")

for archivo in ARCHIVOS:
    if not os.path.exists(archivo):
        print("No encontrada:", archivo)
        continue

    imagen = cv.imread(archivo, cv.IMREAD_GRAYSCALE)

    if imagen is None:
        print("No se pudo leer:", archivo)
        continue

    h, w = imagen.shape
    escala = min(400 / max(h, w), 1.0)

    if escala < 1.0:
        imagen = cv.resize(
            imagen,
            None,
            fx=escala,
            fy=escala,
            interpolation=cv.INTER_AREA
        )

    puntos, descriptores = orb.detectAndCompute(imagen, None)

    if descriptores is None or len(puntos) < MIN_MATCHES:
        print("Sin caracteristicas suficientes:", archivo)
        continue

    # La imagen con sus puntos se prepara una sola vez.
    imagen_visual = cv.drawKeypoints(
        imagen,
        puntos,
        None,
        color=(255, 0, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv.putText(
        imagen_visual,
        archivo,
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    plantillas.append({
        "nombre": archivo,
        "imagen": imagen,
        "puntos": puntos,
        "descriptores": descriptores,
        "visual": imagen_visual
    })

    print(archivo, "| Puntos:", len(puntos))

if not plantillas:
    print("ERROR: No existen plantillas validas.")
    raise SystemExit

print("\nPlantillas cargadas:", len(plantillas))

# =====================================================
# 4. INICIAR CAMARA USB
# =====================================================

camara = cv.VideoCapture(CAMARA_ID, cv.CAP_DSHOW)

if not camara.isOpened():
    camara.release()
    camara = cv.VideoCapture(CAMARA_ID)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la camara.")
    raise SystemExit

# Solicitar MJPG y 60 FPS. La camara puede aceptar o no
# estos valores; se muestran los valores que informa.
camara.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
camara.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
camara.set(cv.CAP_PROP_FPS, 60)
camara.set(cv.CAP_PROP_BUFFERSIZE, 1)

print("\nConfiguracion informada por la camara:")
print("Resolucion:",
      int(camara.get(cv.CAP_PROP_FRAME_WIDTH)),
      "x",
      int(camara.get(cv.CAP_PROP_FRAME_HEIGHT)))
print("FPS solicitados/indicados:", camara.get(cv.CAP_PROP_FPS))

# =====================================================
# 5. CREAR VENTANAS
# =====================================================

cv.namedWindow("DETECTOR ORB", cv.WINDOW_NORMAL)
cv.namedWindow("PUNTOS DE LA CAMARA", cv.WINDOW_NORMAL)
cv.namedWindow("PUNTOS DE LA PLANTILLA", cv.WINDOW_NORMAL)

# =====================================================
# 6. VARIABLES
# =====================================================

poligono = None
puntos_tracking = None
frame_anterior = None
tracking_activo = False

frames_desde_orb = ORB_INTERVALO
contador_frames = 0
fallos_confirmacion_orb = 0
MAX_FALLOS_CONFIRMACION_ORB = 2

ultima_similitud = 0.0
ultima_plantilla = ""
plantilla_actual = plantillas[0]

mejor_inliers = 0
mejores_matches = 0
mejor_similitud = 0.0

imagen_puntos_camara = None

fps = 0.0
tiempo_anterior = time.perf_counter()

# =====================================================
# 7. INICIAR TRACKING
# =====================================================

def iniciar_tracking(gris, poligono_objeto):
    mascara = np.zeros(gris.shape, dtype=np.uint8)

    cv.fillConvexPoly(
        mascara,
        poligono_objeto.reshape(-1, 2),
        255
    )

    return cv.goodFeaturesToTrack(
        gris,
        maxCorners=MAX_PUNTOS,
        qualityLevel=0.01,
        minDistance=5,
        mask=mascara,
        blockSize=7
    )


# =====================================================
# 8. SEGUIMIENTO OPTICO LUCAS-KANADE
# =====================================================

def seguimiento_optico(anterior, actual, puntos, poligono_objeto):
    if puntos is None or len(puntos) < MIN_PUNTOS_TRACKING:
        return None, None

    nuevos, estado, error = cv.calcOpticalFlowPyrLK(
        anterior,
        actual,
        puntos,
        None,
        winSize=(17, 17),
        maxLevel=2,
        criteria=(
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
            15,
            0.03
        )
    )

    if nuevos is None or estado is None:
        return None, None

    validos = estado.reshape(-1) == 1
    origen = puntos[validos]
    destino = nuevos[validos]

    if len(origen) < MIN_PUNTOS_TRACKING:
        return None, None

    transformacion, mascara = cv.estimateAffinePartial2D(
        origen,
        destino,
        method=cv.RANSAC,
        ransacReprojThreshold=3.0
    )

    if transformacion is None or mascara is None:
        return None, None

    inliers_tracking = mascara.reshape(-1).astype(bool)

    if np.sum(inliers_tracking) < MIN_PUNTOS_TRACKING:
        return None, None

    nuevo_poligono = cv.transform(
        poligono_objeto.astype(np.float32),
        transformacion
    )
    nuevo_poligono = np.int32(nuevo_poligono)

    area = abs(cv.contourArea(nuevo_poligono))

    if area < 100 or area > actual.shape[0] * actual.shape[1] * 0.85:
        return None, None

    if not cv.isContourConvex(nuevo_poligono):
        return None, None

    nuevos_puntos = destino[inliers_tracking].reshape(-1, 1, 2)

    return nuevo_poligono, nuevos_puntos


# =====================================================
# 9. VALIDAR LA GEOMETRIA DE UNA DETECCION
# =====================================================

def geometria_valida(plantilla, puntos_plantilla, puntos_frame,
                     buenas, mascara_inliers, poligono, forma_frame):
    """Rechaza homografias con puntos agrupados o poligonos absurdos."""

    alto_frame, ancho_frame = forma_frame[:2]
    alto_plantilla, ancho_plantilla = plantilla["imagen"].shape

    mascara = mascara_inliers.reshape(-1).astype(bool)
    matches_inliers = [m for m, valido in zip(buenas, mascara) if valido]

    if len(matches_inliers) < MIN_INLIERS:
        return False

    # Los puntos validos deben cubrir una zona apreciable de la plantilla.
    puntos_src = np.float32([
        puntos_plantilla[m.queryIdx].pt for m in matches_inliers
    ])

    extension_x = np.ptp(puntos_src[:, 0]) / max(ancho_plantilla, 1)
    extension_y = np.ptp(puntos_src[:, 1]) / max(alto_plantilla, 1)

    if extension_x < MIN_EXTENSION_PLANTILLA:
        return False
    if extension_y < MIN_EXTENSION_PLANTILLA:
        return False

    # El poligono debe estar dentro de la imagen, sin coordenadas extremas.
    puntos_poly = poligono.reshape(-1, 2).astype(np.float32)

    if not np.isfinite(puntos_poly).all():
        return False

    if (
        np.any(puntos_poly[:, 0] < -MARGEN_FRAME)
        or np.any(puntos_poly[:, 1] < -MARGEN_FRAME)
        or np.any(puntos_poly[:, 0] > ancho_frame + MARGEN_FRAME)
        or np.any(puntos_poly[:, 1] > alto_frame + MARGEN_FRAME)
    ):
        return False

    area = abs(cv.contourArea(puntos_poly))
    area_frame = alto_frame * ancho_frame

    if area < area_frame * MIN_AREA_OBJETO:
        return False
    if area > area_frame * MAX_AREA_OBJETO:
        return False

    if not cv.isContourConvex(np.int32(puntos_poly)):
        return False

    # Evitar cuadrilateros extremadamente estrechos o degenerados.
    x, y, w, h = cv.boundingRect(np.int32(puntos_poly))
    if w < 15 or h < 15:
        return False
    if w / max(h, 1) > 4.0 or h / max(w, 1) > 4.0:
        return False

    return True


# =====================================================
# 10. RECONOCIMIENTO ORB
# =====================================================

def reconocer_orb(gris, puntos_frame, descriptores_frame):
    mejor = {
        "poligono": None,
        "inliers": 0,
        "similitud": 0.0,
        "matches": 0,
        "plantilla": None,
        "candidato_visual": None
    }

    if descriptores_frame is None or len(puntos_frame) < MIN_MATCHES:
        return mejor

    candidatos = []

    # Comparar las plantillas solo cuando toca ejecutar ORB.
    for plantilla in plantillas:
        pares = matcher.knnMatch(
            plantilla["descriptores"],
            descriptores_frame,
            k=2
        )

        buenas = [
            m for par in pares
            if len(par) == 2
            for m, n in [par]
            if m.distance < RATIO_TEST * n.distance
        ]

        if len(buenas) >= MIN_MATCHES:
            candidatos.append((len(buenas), plantilla, buenas))

    candidatos.sort(key=lambda x: x[0], reverse=True)

    # La plantilla visual solo cambia si una candidata supera
    # tambien la validacion geometrica completa.

    # Validar solo los dos candidatos con mas coincidencias.
    for cantidad, plantilla, buenas in candidatos[:2]:
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
            3.0
        )

        if H is None or mascara is None:
            continue

        inliers = int(np.sum(mascara))
        similitud = (inliers / len(buenas)) * 100.0

        if inliers < MIN_INLIERS or similitud < MIN_SIMILITUD:
            continue

        # Descartar detecciones cuyos inliers no describen la pieza
        # de forma espacialmente distribuida y geometricamente valida.
        th, tw = plantilla["imagen"].shape

        esquinas = np.float32([
            [0, 0],
            [tw, 0],
            [tw, th],
            [0, th]
        ]).reshape(-1, 1, 2)

        transformadas = cv.perspectiveTransform(esquinas, H)
        candidato_poligono = np.int32(transformadas)

        if not geometria_valida(
            plantilla,
            plantilla["puntos"],
            puntos_frame,
            buenas,
            mascara,
            candidato_poligono,
            gris.shape
        ):
            continue

        if inliers > mejor["inliers"]:
            mejor["candidato_visual"] = plantilla
            mejor.update({
                "poligono": candidato_poligono,
                "inliers": inliers,
                "similitud": similitud,
                "matches": len(buenas),
                "plantilla": plantilla
            })

    return mejor


# =====================================================
# 10. BUCLE PRINCIPAL
# =====================================================

print("\nDETECTOR INICIADO\n")
print("Presiona ESC para salir.\n")

while True:
    ret, frame_original = camara.read()

    if not ret:
        print("No se pudo recibir imagen de la camara.")
        break

    contador_frames += 1

    # Reducir resolucion antes de procesar.
    h_original, w_original = frame_original.shape[:2]
    escala = ANCHO_PROCESO / float(w_original)
    alto = int(h_original * escala)

    frame = cv.resize(
        frame_original,
        (ANCHO_PROCESO, alto),
        interpolation=cv.INTER_AREA
    )

    gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # -------------------------------------------------
    # 10.1 SEGUIR LA PIEZA EN CADA FOTOGRAMA
    # -------------------------------------------------

    tracking_correcto = False

    if (
        tracking_activo
        and frame_anterior is not None
        and poligono is not None
        and puntos_tracking is not None
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
            tracking_correcto = True
        else:
            tracking_activo = False
            puntos_tracking = None

    # -------------------------------------------------
    # 10.2 DECIDIR CUANDO EJECUTAR ORB
    # -------------------------------------------------

    ejecutar_orb = (
        not tracking_activo
        or not tracking_correcto
        or frames_desde_orb >= ORB_INTERVALO
    )

    deteccion_orb = False
    mejor_similitud = 0.0
    mejor_inliers_actual = 0
    mejores_matches_actual = 0

    if ejecutar_orb:
        puntos_frame, descriptores_frame = orb.detectAndCompute(
            gris,
            None
        )

        resultado = reconocer_orb(
            gris,
            puntos_frame,
            descriptores_frame
        )

        frames_desde_orb = 0

        if resultado["candidato_visual"] is not None:
            plantilla_actual = resultado["candidato_visual"]

        if resultado["poligono"] is not None:
            fallos_confirmacion_orb = 0
            deteccion_orb = True
            poligono = resultado["poligono"]

            mejor_similitud = resultado["similitud"]
            mejor_inliers_actual = resultado["inliers"]
            mejores_matches_actual = resultado["matches"]

            ultima_similitud = mejor_similitud
            ultima_plantilla = resultado["plantilla"]["nombre"]

            puntos_tracking = iniciar_tracking(gris, poligono)
            tracking_activo = (
                puntos_tracking is not None
                and len(puntos_tracking) >= MIN_PUNTOS_TRACKING
            )

        elif tracking_correcto:
            # Una comprobacion ORB fallida no cancela inmediatamente
            # un seguimiento optico coherente, pero limita su duracion.
            fallos_confirmacion_orb += 1
            if fallos_confirmacion_orb >= MAX_FALLOS_CONFIRMACION_ORB:
                tracking_activo = False
                puntos_tracking = None
                poligono = None
                HISTORIAL.clear()

        # Actualizar puntos ORB visibles solo cuando se ejecuta ORB.
        imagen_puntos_camara = cv.drawKeypoints(
            frame,
            puntos_frame,
            None,
            color=(0, 255, 0),
            flags=cv.DRAW_MATCHES_FLAGS_DEFAULT
        )

        cv.putText(
            imagen_puntos_camara,
            "Puntos ORB de la camara",
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    else:
        frames_desde_orb += 1

    # -------------------------------------------------
    # 10.3 PUNTOS VISUALES LIGEROS DE LA CAMARA
    # -------------------------------------------------

    if contador_frames % INTERVALO_PUNTOS_VISUALES == 0:
        puntos_visuales = orb_visual.detect(gris, None)

        imagen_puntos_camara = cv.drawKeypoints(
            frame,
            puntos_visuales,
            None,
            color=(0, 255, 0),
            flags=cv.DRAW_MATCHES_FLAGS_DEFAULT
        )

        cv.putText(
            imagen_puntos_camara,
            "Puntos caracteristicos de la camara",
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # -------------------------------------------------
    # 10.4 ESTADO E HISTORIAL
    # -------------------------------------------------

    deteccion_tracking = tracking_correcto and tracking_activo

    HISTORIAL.append(deteccion_orb or deteccion_tracking)

    objeto_encontrado = (
        sum(HISTORIAL) >= MIN_DETECCIONES
        and poligono is not None
    )

    if deteccion_orb:
        mejor_inliers = mejor_inliers_actual
        mejores_matches = mejores_matches_actual
        mejor_similitud = mejor_similitud
    elif deteccion_tracking:
        # Se conservan los datos de la ultima validacion ORB.
        mejor_inliers = mejor_inliers

    # -------------------------------------------------
    # 10.5 MOSTRAR PLANTILLA
    # -------------------------------------------------

    imagen_puntos_plantilla = plantilla_actual["visual"].copy()

    # -------------------------------------------------
    # 10.6 MOSTRAR DETECCION
    # -------------------------------------------------

    if objeto_encontrado:
        estado = "Objeto encontrado"
        color = (0, 255, 0)

        cv.polylines(
            frame,
            [poligono],
            True,
            color,
            2
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

    cv.rectangle(frame, (10, 10), (390, 50), (25, 25, 25), -1)
    cv.putText(
        frame,
        estado,
        (20, 38),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    cv.rectangle(frame, (10, 60), (390, 100), (25, 25, 25), -1)
    cv.putText(
        frame,
        f"Similitud: {similitud_mostrar:.1f}%",
        (20, 88),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

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

    # -------------------------------------------------
    # 10.7 CALCULAR FPS
    # -------------------------------------------------

    ahora = time.perf_counter()
    fps_actual = 1.0 / max(ahora - tiempo_anterior, 0.001)

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

    # -------------------------------------------------
    # 10.8 MOSTRAR VENTANAS
    # -------------------------------------------------

    cv.imshow("DETECTOR ORB", frame)

    if imagen_puntos_camara is not None:
        cv.imshow("PUNTOS DE LA CAMARA", imagen_puntos_camara)

    cv.imshow("PUNTOS DE LA PLANTILLA", imagen_puntos_plantilla)

    # La imagen gris actual se reutiliza como anterior.
    frame_anterior = gris

    if cv.waitKey(1) & 0xFF == 27:
        break

# =====================================================
# 11. CERRAR PROGRAMA
# =====================================================

camara.release()
cv.destroyAllWindows()

print("\nDETECTOR FINALIZADO")
