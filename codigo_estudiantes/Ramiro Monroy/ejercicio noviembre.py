import cv2
import numpy as np
import os
import time
from collections import Counter


# ============================================================
# 1. CONFIGURACION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCHIVOS = {
    "PIEZA A": os.path.join(BASE_DIR, "pieza_A.jpg"),
    "PIEZA B": os.path.join(BASE_DIR, "pieza_B.jpg"),
    "PIEZA C": os.path.join(BASE_DIR, "pieza_C.jpg"),
}

ANCHO_CAMARA = 1280
ALTO_CAMARA = 720

AREA_MINIMA = 1200
AREA_MAXIMA = 250000
MAX_PIEZAS = 3

NFEATURES = 2500
RATIO_TEST = 0.78

MINIMO_COINCIDENCIAS = 8
MINIMO_INLIERS = 5
MINIMA_PROPORCION = 0.30

UMBRAL_FONDO = 22
UMBRAL_ADAPTATIVO = 15

UMBRAL_SUPRESION_CAJA = 0.55
UMBRAL_SUPRESION_CONTENCION = 0.70

# Seguimiento Lucas-Kanade
MAX_PUNTOS_LK = 100
CALIDAD_LK = 0.01
DISTANCIA_LK = 7
MINIMO_PUNTOS_LK = 4
MAXIMO_MISS_LK = 8
MAXIMO_CLASIFICACIONES_POR_TRACK = 12
FRAMES_ESTABLES_PARA_CONTEO = 5


# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================

orb = cv2.ORB_create(
    nfeatures=NFEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    fastThreshold=8,
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


def procesar_gris(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    gris = cv2.GaussianBlur(
        gris,
        (3, 3),
        0,
    )

    gris = clahe.apply(gris)

    return gris


# ============================================================
# 3. CARGAR PLANTILLAS ORB
# ============================================================

plantillas = {}

print("\n========== CARGANDO PLANTILLAS ==========\n")

for nombre, ruta in ARCHIVOS.items():
    imagen = cv2.imread(ruta)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo cargar la plantilla: {ruta}"
        )

    gris = procesar_gris(imagen)

    puntos, descriptores = orb.detectAndCompute(
        gris,
        None,
    )

    if descriptores is None or len(puntos) < 4:
        raise ValueError(
            f"La plantilla {nombre} no tiene suficientes "
            "puntos ORB."
        )

    plantillas[nombre] = {
        "imagen": imagen,
        "gris": gris,
        "puntos": puntos,
        "descriptores": descriptores,
        "alto": gris.shape[0],
        "ancho": gris.shape[1],
    }

    print(
        f"{nombre}: {len(puntos)} puntos ORB"
    )

print("\nPlantillas cargadas correctamente.\n")


# ============================================================
# 4. RECONOCIMIENTO ORB + LOWE + RANSAC
# ============================================================

def reconocer_pieza(imagen_candidata):
    puntos_candidato, desc_candidato = orb.detectAndCompute(
        imagen_candidata,
        None,
    )

    if desc_candidato is None or len(puntos_candidato) < 4:
        return "NO EVALUABLE", 0, 0, 0.0, None

    candidatos_validos = []
    mejor_poligono_global = None

    for nombre, plantilla in plantillas.items():
        pares = matcher.knnMatch(
            plantilla["descriptores"],
            desc_candidato,
            k=2,
        )

        buenas = []

        for par in pares:
            if len(par) < 2:
                continue

            m, n = par

            if m.distance < RATIO_TEST * n.distance:
                buenas.append(m)

        coincidencias = len(buenas)

        if coincidencias < MINIMO_COINCIDENCIAS:
            continue

        origen = np.float32([
            plantilla["puntos"][m.queryIdx].pt
            for m in buenas
        ]).reshape(-1, 1, 2)

        destino = np.float32([
            puntos_candidato[m.trainIdx].pt
            for m in buenas
        ]).reshape(-1, 1, 2)

        if len(origen) < 4:
            continue

        homografia, mascara_ransac = cv2.findHomography(
            origen,
            destino,
            cv2.RANSAC,
            5.0,
        )

        if homografia is None or mascara_ransac is None:
            continue

        inliers = int(mascara_ransac.ravel().sum())
        proporcion = inliers / max(coincidencias, 1)

        if inliers < MINIMO_INLIERS:
            continue

        if proporcion < MINIMA_PROPORCION:
            continue

        alto_plantilla = plantilla["alto"]
        ancho_plantilla = plantilla["ancho"]

        esquinas = np.float32([
            [0, 0],
            [ancho_plantilla - 1, 0],
            [ancho_plantilla - 1, alto_plantilla - 1],
            [0, alto_plantilla - 1],
        ]).reshape(-1, 1, 2)

        poligono = cv2.perspectiveTransform(
            esquinas,
            homografia,
        )

        alto_candidato, ancho_candidato = imagen_candidata.shape[:2]
        area_candidato = alto_candidato * ancho_candidato
        area_poligono = abs(cv2.contourArea(poligono))

        if area_poligono < 100:
            continue

        if area_poligono > area_candidato * 2.5:
            continue

        candidatos_validos.append({
            "nombre": nombre,
            "coincidencias": coincidencias,
            "inliers": inliers,
            "proporcion": proporcion,
            "poligono": poligono,
        })

    if not candidatos_validos:
        return "DESCONOCIDA", 0, 0, 0.0, None

    candidatos_validos.sort(
        key=lambda resultado: (
            resultado["inliers"],
            resultado["coincidencias"],
            resultado["proporcion"],
        ),
        reverse=True,
    )

    mejor = candidatos_validos[0]

    if len(candidatos_validos) > 1:
        segundo = candidatos_validos[1]
        diferencia_inliers = (
            mejor["inliers"] - segundo["inliers"]
        )
        diferencia_coincidencias = (
            mejor["coincidencias"] -
            segundo["coincidencias"]
        )
        if diferencia_inliers < 3 and diferencia_coincidencias < 4:
            return "DESCONOCIDA", mejor["coincidencias"], \
                   mejor["inliers"], mejor["proporcion"], None

    mejor_poligono_global = mejor["poligono"]

    return (
        mejor["nombre"],
        mejor["coincidencias"],
        mejor["inliers"],
        mejor["proporcion"],
        mejor_poligono_global,
    )


# ============================================================
# 5. DETECCION DE OBJETOS
# ============================================================

def obtener_mascara(frame, fondo):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris_suave = cv2.GaussianBlur(
        gris,
        (5, 5),
        0,
    )
    fondo_suave = cv2.GaussianBlur(
        fondo,
        (5, 5),
        0,
    )
    diferencia = cv2.absdiff(
        fondo_suave,
        gris_suave,
    )
    _, mascara_fondo = cv2.threshold(
        diferencia,
        UMBRAL_FONDO,
        255,
        cv2.THRESH_BINARY,
    )
    kernel_apertura = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    mascara = cv2.morphologyEx(
        mascara_fondo,
        cv2.MORPH_OPEN,
        kernel_apertura,
        iterations=1,
    )
    kernel_cierre = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel_cierre,
        iterations=1,
    )
    kernel_dilatacion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    mascara = cv2.dilate(
        mascara,
        kernel_dilatacion,
        iterations=1,
    )
    numero, etiquetas, estadisticas, _ = \
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8,
        )
    mascara_limpia = np.zeros_like(mascara)
    for indice in range(1, numero):
        x, y, w, h, area = estadisticas[indice]
        if area < 300:
            continue
        mascara_limpia[etiquetas == indice] = 255
    return mascara_limpia


def interseccion_cajas(caja_a, caja_b):
    ax, ay, aw, ah = caja_a
    bx, by, bw, bh = caja_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def calcular_iou(caja_a, caja_b):
    interseccion = interseccion_cajas(caja_a, caja_b)

    ax, ay, aw, ah = caja_a
    bx, by, bw, bh = caja_b

    union = aw * ah + bw * bh - interseccion

    if union <= 0:
        return 0.0

    return interseccion / union


def suprimir_cajas_solapadas(
        cajas,
        umbral_iou=UMBRAL_SUPRESION_CAJA):
    seleccionadas = []
    cajas_ordenadas = sorted(
        cajas,
        key=lambda caja: caja[2] * caja[3],
        reverse=True,
    )
    for caja in cajas_ordenadas:
        solapada = False
        for existente in seleccionadas:
            interseccion = interseccion_cajas(caja, existente)
            area_caja = caja[2] * caja[3]
            area_existente = existente[2] * existente[3]
            proporcion_caja = interseccion / max(area_caja, 1)
            proporcion_existente = interseccion / max(area_existente, 1)

            if (
                calcular_iou(caja, existente) >= umbral_iou
                or proporcion_caja >= UMBRAL_SUPRESION_CONTENCION
                or proporcion_existente >= UMBRAL_SUPRESION_CONTENCION
            ):
                solapada = True
                break
        if not solapada:
            seleccionadas.append(caja)
    seleccionadas.sort(key=lambda caja: caja[0])
    return seleccionadas


def detectar_piezas(mascara):
    cajas = []
    numero, _, estadisticas, _ = cv2.connectedComponentsWithStats(
        mascara,
        connectivity=8,
    )

    candidatos = []

    for indice in range(1, numero):
        x, y, w, h, area = estadisticas[indice]

        if area < AREA_MINIMA or area > AREA_MAXIMA:
            continue

        if w < 20 or h < 20:
            continue

        proporcion = w / max(h, 1)

        if proporcion < 0.15 or proporcion > 6.5:
            continue

        candidatos.append((x, y, w, h, area))

    candidatos.sort(
        key=lambda candidato: candidato[4],
        reverse=True,
    )

    for x, y, w, h, _ in candidatos:
        caja = (x, y, w, h)
        solapada = False

        for existente in cajas:
            interseccion = interseccion_cajas(caja, existente)
            area_caja = w * h
            area_existente = existente[2] * existente[3]
            proporcion_caja = interseccion / max(area_caja, 1)
            proporcion_existente = interseccion / max(area_existente, 1)

            if (
                calcular_iou(caja, existente) >= UMBRAL_SUPRESION_CAJA
                or proporcion_caja >= UMBRAL_SUPRESION_CONTENCION
                or proporcion_existente >= UMBRAL_SUPRESION_CONTENCION
            ):
                solapada = True
                break

        if solapada:
            continue

        cajas.append(caja)
        if len(cajas) >= MAX_PIEZAS:
            break

    return cajas[:MAX_PIEZAS]


# ============================================================
# 6. LUCAS-KANADE
# ============================================================

def inicializar_puntos_lk(gris, caja):
    x, y, w, h = caja

    alto, ancho = gris.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(ancho, x + w)
    y2 = min(alto, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    mascara = np.zeros(
        gris.shape,
        dtype=np.uint8,
    )

    mascara[y1:y2, x1:x2] = 255

    puntos = cv2.goodFeaturesToTrack(
        gris,
        maxCorners=MAX_PUNTOS_LK,
        qualityLevel=CALIDAD_LK,
        minDistance=DISTANCIA_LK,
        mask=mascara,
        blockSize=7,
    )

    return puntos


def actualizar_lk(gris_anterior, gris_actual, puntos):
    if puntos is None or len(puntos) < MINIMO_PUNTOS_LK:
        return None, None

    nuevos, estado, _ = cv2.calcOpticalFlowPyrLK(
        gris_anterior,
        gris_actual,
        puntos,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        ),
    )

    if nuevos is None or estado is None:
        return None, None

    invertidos, estado_inversa, _ = cv2.calcOpticalFlowPyrLK(
        gris_actual,
        gris_anterior,
        nuevos,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        ),
    )

    if invertidos is None or estado_inversa is None:
        buenos_nuevos = nuevos[estado.ravel() == 1]
        buenos_anteriores = puntos[estado.ravel() == 1]
    else:
        estado_final = (
            (estado.ravel() == 1) &
            (estado_inversa.ravel() == 1)
        )

        diferencias = np.linalg.norm(
            invertidos - puntos,
            axis=2,
        )

        estado_final = estado_final & (diferencias[:, 0] < 1.5)

        buenos_nuevos = nuevos[estado_final]
        buenos_anteriores = puntos[estado_final]

    if len(buenos_nuevos) < MINIMO_PUNTOS_LK:
        return None, None

    alto, ancho = gris_actual.shape[:2]

    dentro_imagen = (
        (buenos_nuevos[:, 0, 0] >= 0) &
        (buenos_nuevos[:, 0, 0] < ancho) &
        (buenos_nuevos[:, 0, 1] >= 0) &
        (buenos_nuevos[:, 0, 1] < alto)
    )

    buenos_nuevos = buenos_nuevos[dentro_imagen]
    buenos_anteriores = buenos_anteriores[dentro_imagen]

    if len(buenos_nuevos) < MINIMO_PUNTOS_LK:
        return None, None

    return buenos_anteriores, buenos_nuevos


def caja_desde_puntos(puntos, alto, ancho):
    if puntos is None or len(puntos) == 0:
        return None

    puntos = puntos.reshape(-1, 2)

    x_min = max(0, int(np.min(puntos[:, 0])))
    y_min = max(0, int(np.min(puntos[:, 1])))
    x_max = min(ancho, int(np.max(puntos[:, 0])))
    y_max = min(alto, int(np.max(puntos[:, 1])))

    if x_max <= x_min or y_max <= y_min:
        return None

    return (
        x_min,
        y_min,
        x_max - x_min,
        y_max - y_min,
    )


# ============================================================
# 7. ACTIVAR CAMARA
# ============================================================

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camara.isOpened():
    camara = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

camara.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    ANCHO_CAMARA,
)

camara.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    ALTO_CAMARA,
)

print("Camara activada.")
print("Deja la banda completamente vacia.")
print("Presiona ESPACIO para capturar el fondo.")
print("Presiona ESC para salir.\n")

fondo = None

while True:
    correcto, frame = camara.read()

    if not correcto:
        print("No se pudo recibir imagen.")
        break

    cv2.putText(
        frame,
        "BANDA VACIA - ESPACIO PARA CAPTURAR FONDO",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow(
        "CONFIGURACION DEL FONDO",
        frame,
    )

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 32:
        capturas_fondo = []

        for _ in range(21):
            ok, captura = camara.read()

            if not ok:
                continue

            gris = cv2.cvtColor(
                captura,
                cv2.COLOR_BGR2GRAY,
            )

            gris = cv2.GaussianBlur(
                gris,
                (5, 5),
                0,
            )

            capturas_fondo.append(gris)

        if not capturas_fondo:
            raise RuntimeError(
                "No se pudieron capturar fotogramas del fondo."
            )

        fondo = np.median(
            np.array(capturas_fondo),
            axis=0,
        ).astype(np.uint8)

        print("Fondo capturado correctamente.")
        cv2.destroyWindow("CONFIGURACION DEL FONDO")
        break

    if tecla == 27:
        camara.release()
        cv2.destroyAllWindows()
        raise SystemExit


# ============================================================
# 8. SEGUIMIENTO MULTI-OBJETO Y CONTADORES
# ============================================================

tracks = {}
siguiente_id = 1

contadores = {
    "PIEZA A": 0,
    "PIEZA B": 0,
    "PIEZA C": 0,
    "DESCONOCIDA": 0,
}


def asociar_cajas(cajas, tracks, ancho, alto):
    asignaciones = {}
    tracks_usados = set()
    cajas_usadas = set()

    candidatos = []

    for track_id, track in tracks.items():
        for indice, caja in enumerate(cajas):
            iou = calcular_iou(
                track["caja"],
                caja,
            )

            cx_track = track["caja"][0] + track["caja"][2] / 2
            cy_track = track["caja"][1] + track["caja"][3] / 2

            cx_caja = caja[0] + caja[2] / 2
            cy_caja = caja[1] + caja[3] / 2

            distancia = np.hypot(
                cx_track - cx_caja,
                cy_track - cy_caja,
            )

            escala = max(
                40,
                max(track["caja"][2], track["caja"][3]) * 0.6,
            )

            if iou >= 0.12 or distancia <= escala:
                costo = distancia / escala - iou
                candidatos.append((costo, track_id, indice))

    candidatos.sort(key=lambda elemento: elemento[0])

    for _, track_id, indice_caja in candidatos:
        if track_id in tracks_usados:
            continue

        if indice_caja in cajas_usadas:
            continue

        asignaciones[track_id] = indice_caja
        tracks_usados.add(track_id)
        cajas_usadas.add(indice_caja)

    nuevas_cajas = [
        (indice, caja)
        for indice, caja in enumerate(cajas)
        if indice not in cajas_usadas
    ]

    return asignaciones, nuevas_cajas


gris_anterior = None
tiempo_anterior = time.time()

print("\nSistema de inspeccion iniciado.\n")


# ============================================================
# 9. CICLO PRINCIPAL
# ============================================================

while True:
    correcto, frame = camara.read()

    if not correcto:
        print("Error al capturar fotograma.")
        break

    alto, ancho = frame.shape[:2]

    gris_actual = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gris_suave = cv2.GaussianBlur(
        gris_actual,
        (5, 5),
        0,
    )

    mascara = obtener_mascara(
        frame,
        fondo,
    )

    cajas = detectar_piezas(mascara)

    # Actualizar cada seguimiento Lucas-Kanade individualmente.
    for track_id, track in tracks.items():
        if gris_anterior is not None and track["puntos"] is not None:
            _, nuevos = actualizar_lk(
                gris_anterior,
                gris_suave,
                track["puntos"],
            )
            if nuevos is not None:
                track["puntos"] = nuevos.reshape(-1, 1, 2)
                caja_lk = caja_desde_puntos(
                    track["puntos"],
                    alto,
                    ancho,
                )
                if caja_lk is not None:
                    track["caja_lk"] = caja_lk
                    track["misses"] = 0
            else:
                track["puntos"] = None
                track["caja_lk"] = None
                track["misses"] += 1
        for punto in (
            track["puntos"]
            if track["puntos"] is not None
            else []
        ):
            px, py = punto.ravel().astype(int)

            cv2.circle(
                frame,
                (px, py),
                2,
                (255, 255, 0),
                -1,
            )

    asignaciones, nuevas_cajas = asociar_cajas(
        cajas,
        tracks,
        ancho,
        alto,
    )

    for track_id, indice_caja in asignaciones.items():
        track = tracks[track_id]
        caja = cajas[indice_caja]
        track["caja"] = caja
        track["caja_lk"] = caja
        track["misses"] = 0
        track["age"] += 1
        track["frames_desde_reinicio"] += 1
        if (
            track["frames_desde_reinicio"] >= 8
            or track["puntos"] is None
        ):
            track["puntos"] = inicializar_puntos_lk(
                gris_suave,
                caja,
            )
            track["frames_desde_reinicio"] = 0

    tracks_asignados = set(asignaciones.keys())

    for track_id, track in tracks.items():
        if track_id not in tracks_asignados:
            track["misses"] += 1

    espacios_disponibles = max(0, MAX_PIEZAS - len(tracks))

    for _, caja in nuevas_cajas[:espacios_disponibles]:
        puntos = inicializar_puntos_lk(
            gris_suave,
            caja,
        )

        tracks[siguiente_id] = {
            "caja": caja,
            "caja_lk": caja,
            "puntos": puntos,
            "misses": 0,
            "age": 1,
            "frames_desde_reinicio": 0,
            "votos": Counter(),
            "nombre_estable": None,
            "frames_estables": 0,
            "contado": False,
            "frames_desde_reconocimiento": 4,
            "nombre_actual": "DESCONOCIDA",
        }

        siguiente_id += 1

    tracks_a_eliminar = [
        track_id
        for track_id, track in tracks.items()
        if track["misses"] >= MAXIMO_MISS_LK
    ]

    for track_id in tracks_a_eliminar:
        del tracks[track_id]

    cantidad_objetos = 0

    for track_id, track in tracks.items():
        x, y, w, h = track["caja"]
        margen_x = int(w * 0.18)
        margen_y = int(h * 0.18)
        x1 = max(0, x - margen_x)
        y1 = max(0, y - margen_y)
        x2 = min(ancho, x + w + margen_x)
        y2 = min(alto, y + h + margen_y)
        recorte = frame[y1:y2, x1:x2]
        if recorte.size == 0:
            continue

        track["frames_desde_reconocimiento"] += 1
        debe_reconocer = (
            track["frames_desde_reconocimiento"] >= 4
        )

        if debe_reconocer:
            gris_candidato = procesar_gris(recorte)
            (
                nombre,
                coincidencias,
                inliers,
                proporcion,
                poligono,
            ) = reconocer_pieza(gris_candidato)
            track["frames_desde_reconocimiento"] = 0
            track["nombre_actual"] = nombre
        else:
            nombre = track["nombre_actual"]
            coincidencias = 0
            inliers = 0
            proporcion = 0.0
            poligono = None
        track["votos"][nombre] += 1
        votos_totales = sum(track["votos"].values())
        if votos_totales > MAXIMO_CLASIFICACIONES_POR_TRACK:
            nombre_mas_frecuente = track["votos"].most_common(1)[0][0]
            track["votos"] = Counter({
                nombre_mas_frecuente:
                track["votos"][nombre_mas_frecuente]
            })
        nombre_mas_frecuente, frecuencia = \
            track["votos"].most_common(1)[0]
        if frecuencia >= FRAMES_ESTABLES_PARA_CONTEO:
            track["nombre_estable"] = nombre_mas_frecuente
            track["frames_estables"] = frecuencia
        else:
            track["nombre_estable"] = None
            track["frames_estables"] = 0

        if nombre == "NO EVALUABLE":
            color = (0, 255, 255)
        elif nombre == "DESCONOCIDA":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        etiqueta = nombre

        cv2.putText(
            frame,
            etiqueta,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"M:{coincidencias} I:{inliers}",
            (x1, min(alto - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        if poligono is not None:
            poligono_original = poligono.copy()
            poligono_original[:, 0, 0] += x1
            poligono_original[:, 0, 1] += y1

            cv2.polylines(
                frame,
                [np.int32(poligono_original)],
                True,
                (255, 0, 255),
                2,
            )

        if track["nombre_estable"] is not None and not track["contado"]:
            if track["nombre_estable"] in (
                "PIEZA A",
                "PIEZA B",
                "PIEZA C",
            ):
                contadores[track["nombre_estable"]] += 1
                track["contado"] = True

            elif track["nombre_estable"] in (
                "DESCONOCIDA",
                "NO EVALUABLE",
            ):
                contadores["DESCONOCIDA"] += 1
                track["contado"] = True

        cv2.putText(
            frame,
            f"TRACK {track_id}",
            (x1, min(alto - 10, y2 + 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        cantidad_objetos += 1

    cantidad_lk = sum(
        0 if track["puntos"] is None else len(track["puntos"])
        for track in tracks.values()
    )

    # --------------------------------------------------------
    # PANEL DE INFORMACION
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (420, 190),
        (30, 30, 30),
        -1,
    )

    cv2.putText(
        frame,
        f"OBJETOS DETECTADOS: {cantidad_objetos}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"ORB: {NFEATURES}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"PUNTOS LUCAS-KANADE: {cantidad_lk}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"A: {contadores['PIEZA A']}  "
        f"B: {contadores['PIEZA B']}  "
        f"C: {contadores['PIEZA C']}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"DESCONOCIDAS: {contadores['DESCONOCIDA']}  "
        f"TRACKS ACTIVOS: {len(tracks)}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    tiempo_actual = time.time()
    diferencia = tiempo_actual - tiempo_anterior
    fps = 1.0 / diferencia if diferencia > 0 else 0.0
    tiempo_anterior = tiempo_actual

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (ancho - 130, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "SISTEMA DE INSPECCION ORB + RANSAC + LUCAS-KANADE",
        frame,
    )

    gris_anterior = gris_suave.copy()

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break


# ============================================================
# 10. FINALIZAR
# ============================================================

camara.release()
cv2.destroyAllWindows()

print("\nSistema finalizado.")
print("\nContadores:")

for nombre, cantidad in contadores.items():
    print(f"{nombre}: {cantidad}")