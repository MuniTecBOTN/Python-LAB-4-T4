import cv2
import numpy as np
import time

# ==========================================
# 1. CONFIGURACION
# ==========================================

PLANTILLAS = [
    # OBJETO 1
    {"ruta": "main/plantillas/cap1.jpg", "objeto": "OBJETO 1", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap1_mas_cerca.jpg", "objeto": "OBJETO 1", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap1_lejos.jpg", "objeto": "OBJETO 1", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap1_mas_lejos.jpg", "objeto": "OBJETO 1", "distancia": "LEJOS"},

    # OBJETO 2
    {"ruta": "main/plantillas/cap2.jpg", "objeto": "OBJETO 2", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap2_mas_cerca.jpg", "objeto": "OBJETO 2", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap2_lejos.jpg", "objeto": "OBJETO 2", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap2_mas_lejos.jpg", "objeto": "OBJETO 2", "distancia": "LEJOS"},

    # OBJETO 3
    {"ruta": "main/plantillas/cap3.jpg", "objeto": "OBJETO 3", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap3_mas_cerca.jpg", "objeto": "OBJETO 3", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap3_lejos.jpg", "objeto": "OBJETO 3", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap3_mas_lejos.jpg", "objeto": "OBJETO 3", "distancia": "LEJOS"},

    # OBJETO 4
    {"ruta": "main/plantillas/cap4.jpg", "objeto": "OBJETO 4", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap4_cerca.jpg", "objeto": "OBJETO 4", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap4_mas_lejos.jpg", "objeto": "OBJETO 4", "distancia": "LEJOS"},
    {"ruta": "main/plantillas/cap4_mas_cerca.jpg", "objeto": "OBJETO 4", "distancia": "CERCA"},
]

# Resolución interna optimizada para altos FPS
ANCHO = 480
ALTO = 270

MIN_MATCHES = 12
MIN_INLIERS = 10
MIN_INLIER_RATIO = 0.38
MAX_AREA_RATIO = 0.45  # Límite estricto para evitar que el cuadro verde abarque toda la pantalla
SMOOTHING = 0.25       # Suavizado para mantener cuadros estáticos

# ==========================================
# 2. PREPROCESAMIENTO OPTIMIZADO
# ==========================================

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

orb = cv2.ORB_create(
    nfeatures=600,
    scaleFactor=1.2,
    nlevels=6,
    edgeThreshold=15,
    patchSize=31,
    firstLevel=0,
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ==========================================
# 3. CARGAR PLANTILLAS
# ==========================================

templates = []

for plantilla in PLANTILLAS:
    ruta = plantilla["ruta"]
    nombre = plantilla["objeto"]
    distancia = plantilla["distancia"]

    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape
    if w > 500:
        scale = 500 / w
        img = cv2.resize(img, None, fx=scale, fy=scale)

    img = clahe.apply(img)
    kp, des = orb.detectAndCompute(img, None)

    if des is None or len(kp) < MIN_MATCHES:
        continue

    templates.append({
        "name": nombre,
        "distance": distancia,
        "kp": kp,
        "des": des,
        "shape": img.shape
    })

print(f"Plantillas cargadas exitosamente: {len(templates)}")

# ==========================================
# 4. FUNCIONES AUXILIARES
# ==========================================

def validar_geometria(projected, frame_shape):
    fh, fw = frame_shape[:2]
    if projected is None or not np.isfinite(projected).all():
        return False

    polygon = projected.reshape(4, 2).astype(np.float32)
    if not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
        return False

    area = abs(cv2.contourArea(polygon))
    # Evitar áreas demasiado pequeñas o gigantescas que se salgan de proporción
    if area < 300 or area > fw * fh * MAX_AREA_RATIO:
        return False

    # Validar proporciones de lados para mantener cuadros centrados y bien ajustados
    l_izq = np.linalg.norm(polygon[0] - polygon[1])
    l_der = np.linalg.norm(polygon[3] - polygon[2])
    l_arr = np.linalg.norm(polygon[0] - polygon[3])
    l_aba = np.linalg.norm(polygon[1] - polygon[2])

    if min(l_izq, l_der, l_arr, l_aba) < 15:
        return False

    if max(l_izq, l_der) / (min(l_izq, l_der) + 1e-5) > 1.4 or \
       max(l_arr, l_aba) / (min(l_arr, l_aba) + 1e-5) > 1.4:
        return False

    if (np.max(polygon[:, 0]) < 0 or np.max(polygon[:, 1]) < 0 or
        np.min(polygon[:, 0]) > fw or np.min(polygon[:, 1]) > fh):
        return False

    return True

def calcular_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
        
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

# ==========================================
# 5. CAMARA
# ==========================================

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

fps_counter = 0
fps_start = time.perf_counter()
fps = 0.0

historial_poligonos = {}
ausencias = {}

# ==========================================
# 6. BUCLE PRINCIPAL DE DETECCION
# ==========================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_h, original_w = frame.shape[:2]
    small = cv2.resize(frame, (ANCHO, ALTO))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_evaluable = laplacian_var > 4.0

    detecciones_brutas = []
    detecciones_validas = []

    if is_evaluable:
        kp_frame, des_frame = orb.detectAndCompute(gray, None)

        if des_frame is not None and len(des_frame) >= MIN_MATCHES:
            for t in templates:
                matches = matcher.knnMatch(t["des"], des_frame, k=2)
                good_matches = []

                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.72 * n.distance:
                            good_matches.append(m)

                if len(good_matches) < MIN_MATCHES:
                    continue

                src_pts = np.float32([t["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                if matrix is None or mask is None:
                    continue

                inliers = mask.ravel().astype(bool)
                cantidad_inliers = int(np.sum(inliers))
                confianza_actual = (cantidad_inliers / len(good_matches)) * 100

                if cantidad_inliers < MIN_INLIERS or confianza_actual < MIN_INLIER_RATIO * 100:
                    continue

                h_t, w_t = t["shape"]
                corners = np.float32([[0, 0], [0, h_t - 1], [w_t - 1, h_t - 1], [w_t - 1, 0]]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, matrix)

                if not validar_geometria(projected, gray.shape):
                    continue

                poly_pts = projected.reshape(4, 2)
                x, y, bw, bh = cv2.boundingRect(np.int32(poly_pts))
                score = cantidad_inliers * (confianza_actual / 100)

                detecciones_brutas.append({
                    "name": t["name"],
                    "distance": t["distance"],
                    "score": score,
                    "inliers": cantidad_inliers,
                    "confianza": confianza_actual,
                    "polygon": poly_pts,
                    "box": (x, y, bw, bh)
                })

        # NMS para objetos conocidos juntos
        detecciones_brutas = sorted(detecciones_brutas, key=lambda x: x["score"], reverse=True)
        for det in detecciones_brutas:
            keep = True
            for accepted in detecciones_validas:
                if det["name"] == accepted["name"] or calcular_iou(det["box"], accepted["box"]) > 0.15:
                    keep = False
                    break
            if keep:
                detecciones_validas.append(det)

    # ==========================================
    # 7. DETECCION DE OBJETOS DESCONOCIDOS (CON MASCARA DE EXCLUSION)
    # ==========================================
    objetos_desconocidos = []
    if is_evaluable:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Crear máscara para bloquear el interior de los objetos conocidos detectados en verde
        mask_conocidos = np.zeros((ALTO, ANCHO), dtype=np.uint8)
        for det in detecciones_validas:
            cv2.fillPoly(mask_conocidos, [np.int32(det["polygon"])], 255)
        
        # Anular los contornos que caigan dentro del área de los objetos conocidos
        thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(mask_conocidos))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            # Filtro anti-ruido estricto para ignorar detalles pequeños
            if 1800 < area < (ANCHO * ALTO * 0.50):
                rx, ry, rw, rh = cv2.boundingRect(c)
                objetos_desconocidos.append((rx, ry, rw, rh))

    # ==========================================
    # 8. RENDERIZADO VISUAL Y ESTABILIZACION
    # ==========================================
    output = frame.copy()
    scale_x = original_w / ANCHO
    scale_y = original_h / ALTO
    detectados_ahora = set()

    if not is_evaluable:
        cv2.rectangle(output, (10, 10), (320, 45), (25, 25, 25), -1)
        cv2.putText(output, "NITIDEZ BAJA - MEJORAR ILUMINACION", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

    else:
        # 1. Dibujar rectángulos rojos solo para objetos desconocidos externos reales
        for (rx, ry, rw, rh) in objetos_desconocidos:
            rx_s = int(rx * scale_x)
            ry_s = int(ry * scale_y)
            rw_s = int(rw * scale_x)
            rh_s = int(rh * scale_y)

            cv2.rectangle(output, (rx_s, ry_s), (rx_s + rw_s, ry_s + rh_s), (0, 0, 255), 2)
            cv2.putText(
                output,
                "PIEZA NO RECONOCIDA",
                (rx_s, max(ry_s - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                2
            )

        # 2. Dibujar polígonos verdes y etiquetas centradas para objetos conocidos
        if len(detecciones_validas) > 0:
            principal = detecciones_validas[0]

            for det in detecciones_validas:
                nombre = det["name"]
                detectados_ahora.add(nombre)
                ausencias[nombre] = 0

                # Suavizado para centrar y estabilizar el cuadro
                if nombre in historial_poligonos:
                    historial_poligonos[nombre] = SMOOTHING * det["polygon"] + (1 - SMOOTHING) * historial_poligonos[nombre]
                else:
                    historial_poligonos[nombre] = det["polygon"].copy()

                polygon = historial_poligonos[nombre].copy()
                polygon[:, 0] *= scale_x
                polygon[:, 1] *= scale_y
                polygon_int = np.int32(polygon)

                # Polígono verde centrado y ajustado al tamaño real
                cv2.polylines(output, [polygon_int], True, (0, 255, 0), 2, cv2.LINE_AA)

                # Centro del objeto
                x, y, bw, bh = cv2.boundingRect(polygon_int)
                cv2.circle(output, (x + bw // 2, y + bh // 2), 4, (0, 255, 255), -1)

                # Etiqueta arriba del rectángulo verde
                cv2.putText(
                    output,
                    f"{det['name']} ({det['distance']}) - {det['confianza']:.0f}%",
                    (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 0),
                    2
                )

            # Cuadro negro informativo superior izquierdo compacto
            cv2.rectangle(output, (10, 10), (280, 95), (25, 25, 25), -1)
            cv2.putText(output, "OBJETO PRINCIPAL DETECTADO", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
            cv2.putText(output, f"Clasificacion: {principal['name']}", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(output, f"Distancia: {principal['distance']}", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
            cv2.putText(output, f"Confianza: {principal['confianza']:.1f}% | Inliers: {principal['inliers']}", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        elif len(objetos_desconocidos) > 0:
            cv2.rectangle(output, (10, 10), (280, 55), (25, 25, 25), -1)
            cv2.putText(output, "OBJETO NO RECONOCIDO", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 2)
            cv2.putText(output, "No existe coincidencia suficiente", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            cv2.rectangle(output, (10, 10), (280, 55), (25, 25, 25), -1)
            cv2.putText(output, "BUSCANDO OBJETOS...", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 165, 255), 2)
            cv2.putText(output, "Coloque una pieza en el encuadre", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Limpieza de memoria de objetos ausentes
    for nom in list(historial_poligonos.keys()):
        if nom not in detectados_ahora:
            ausencias[nom] = ausencias.get(nom, 0) + 1
            if ausencias[nom] > 4:
                del historial_poligonos[nom]
                del ausencias[nom]

    # ==========================================
    # 9. FPS CONTINUO
    # ==========================================
    fps_counter += 1
    elapsed = time.perf_counter() - fps_start
    if elapsed >= 1.0:
        fps = fps_counter / elapsed
        fps_counter = 0
        fps_start = time.perf_counter()

    cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("DETECCION Y EVALUACION ORB", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()