import cv2
import numpy as np
import time

# ==========================================
# 1. CONFIGURACION
# ==========================================

PLANTILLAS = [
    # OBJETO 1
    {"ruta": "main/plantillas/cap1_cerca.jpg", "objeto": "OBJETO 1", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap1_lejos.jpg", "objeto": "OBJETO 1", "distancia": "LEJOS"},

    # OBJETO 2
    {"ruta": "main/plantillas/cap2_cerca.jpg", "objeto": "OBJETO 2", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap2_lejos.jpg", "objeto": "OBJETO 2", "distancia": "LEJOS"},

    # OBJETO 3
    {"ruta": "main/plantillas/cap3_cerca.jpg", "objeto": "OBJETO 3", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap3_lejos.jpg", "objeto": "OBJETO 3", "distancia": "LEJOS"},

    # OBJETO 4
    {"ruta": "main/plantillas/cap4_cerca.jpg", "objeto": "OBJETO 4", "distancia": "CERCA"},
    {"ruta": "main/plantillas/cap4_lejos.jpg", "objeto": "OBJETO 4", "distancia": "LEJOS"},
]

ANCHO = 640
ALTO = 480

MIN_MATCHES = 8
MIN_INLIERS = 6
MIN_INLIER_RATIO = 0.28
MAX_AREA_RATIO = 0.80

MIN_CONTOUR_AREA = 1800          
MIN_UNKNOWN_AREA = 2500          
IOU_THRESHOLD = 0.35             

# ==========================================
# 2. PREPROCESAMIENTO
# ==========================================

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

orb = cv2.ORB_create(
    nfeatures=1200,
    scaleFactor=1.2,
    nlevels=8,
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
        print("Error al cargar:", ruta)
        raise SystemExit

    h, w = img.shape
    if w > 550:
        scale = 550 / w
        img = cv2.resize(img, None, fx=scale, fy=scale)

    img = clahe.apply(img)
    kp, des = orb.detectAndCompute(img, None)

    if des is None or len(kp) < MIN_MATCHES:
        print(f"Plantilla débil (pocos keypoints): {ruta}")
        continue

    templates.append({
        "name": nombre,
        "distance": distancia,
        "img": img,
        "kp": kp,
        "des": des,
        "shape": img.shape
    })

print(f"Plantillas cargadas: {len(templates)}")

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
    if area < 120 or area > fw * fh * MAX_AREA_RATIO:
        return False

    if (np.max(polygon[:, 0]) < 0 or np.max(polygon[:, 1]) < 0 or
        np.min(polygon[:, 0]) > fw or np.min(polygon[:, 1]) > fh):
        return False

    x, y, w, h = cv2.boundingRect(np.int32(polygon))
    if w < 15 or h < 15:
        return False
    ratio = max(w, h) / (min(w, h) + 1e-5)
    if ratio > 8.0:
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

def nms_detecciones(dets, iou_thresh=0.35):
    if not dets:
        return []

    dets = sorted(dets, key=lambda x: x["score"], reverse=True)
    keep = []

    for det in dets:
        should_keep = True
        for accepted in keep:
            if det["name"] == accepted["name"] and calcular_iou(det["box"], accepted["box"]) > iou_thresh:
                should_keep = False
                break
            # También eliminar solapamientos fuertes entre objetos diferentes
            if calcular_iou(det["box"], accepted["box"]) > 0.55:
                should_keep = False
                break
        if should_keep:
            keep.append(det)
    return keep

# ==========================================
# 5. CAMARA
# ==========================================

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

fps_counter = 0
fps_start = time.perf_counter()
fps = 0.0

# ==========================================
# 6. BUCLE PRINCIPAL
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
    is_evaluable = laplacian_var > 5.0

    detecciones_validas = []
    unknown_boxes = []  

    if is_evaluable:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_CONTOUR_AREA:
                valid_contours.append(cnt)


        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)


            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(ANCHO, x + w + pad)
            y2 = min(ALTO, y + h + pad)

            roi = gray[y1:y2, x1:x2]
            if roi.size < 400:
                continue

            kp_roi, des_roi = orb.detectAndCompute(roi, None)

            if des_roi is None or len(des_roi) < MIN_MATCHES:
                if (x2 - x1) * (y2 - y1) >= MIN_UNKNOWN_AREA:
                    unknown_boxes.append((x1, y1, x2 - x1, y2 - y1))
                continue

            mejor_deteccion = None
            mejor_score = 0

            for t in templates:
                matches = matcher.knnMatch(t["des"], des_roi, k=2)
                good = []

                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.73 * n.distance:
                            good.append(m)

                if len(good) < MIN_MATCHES:
                    continue

                src_pts = np.float32([t["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                dst_pts[:, :, 0] += x1
                dst_pts[:, :, 1] += y1

                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.5)
                if matrix is None or mask is None:
                    continue

                inliers = mask.ravel().astype(bool)
                n_inliers = int(np.sum(inliers))
                confianza = (n_inliers / len(good)) * 100.0

                if n_inliers < MIN_INLIERS or confianza < MIN_INLIER_RATIO * 100:
                    continue

                h_t, w_t = t["shape"]
                corners = np.float32([[0, 0], [0, h_t-1], [w_t-1, h_t-1], [w_t-1, 0]]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, matrix)

                if not validar_geometria(projected, gray.shape):
                    continue

                poly = projected.reshape(4, 2)
                bx, by, bw, bh = cv2.boundingRect(np.int32(poly))
                score = n_inliers * (confianza / 100.0)

                if score > mejor_score:
                    mejor_score = score
                    mejor_deteccion = {
                        "name": t["name"],
                        "distance": t["distance"],
                        "score": score,
                        "inliers": n_inliers,
                        "confianza": confianza,
                        "polygon": poly,
                        "box": (bx, by, bw, bh)
                    }

            if mejor_deteccion is not None:
                detecciones_validas.append(mejor_deteccion)
            else:
                if (x2 - x1) * (y2 - y1) >= MIN_UNKNOWN_AREA:
                    unknown_boxes.append((x1, y1, x2 - x1, y2 - y1))

        detecciones_validas = nms_detecciones(detecciones_validas, IOU_THRESHOLD)

        filtered_unknown = []
        for ub in unknown_boxes:
            overlap = False
            for det in detecciones_validas:
                if calcular_iou(ub, det["box"]) > 0.40:
                    overlap = True
                    break
            if not overlap:
                filtered_unknown.append(ub)
        unknown_boxes = filtered_unknown

    # ==========================================
    # 7. RENDERIZADO
    # ==========================================
    output = frame.copy()
    scale_x = original_w / ANCHO
    scale_y = original_h / ALTO

    if not is_evaluable:
        cv2.rectangle(output, (10, 10), (340, 48), (20, 20, 20), -1)
        cv2.putText(output, "NITIDEZ BAJA - MEJORAR ILUMINACION", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)

    else:
        for det in detecciones_validas:
            poly = det["polygon"].copy()
            poly[:, 0] *= scale_x
            poly[:, 1] *= scale_y
            poly_int = np.int32(poly)

        
            cv2.polylines(output, [poly_int], True, (0, 255, 0), 2, cv2.LINE_AA)

           
            x, y, bw, bh = cv2.boundingRect(poly_int)
            cv2.circle(output, (x + bw // 2, y + bh // 2), 5, (0, 255, 255), -1)

         
            label = f"{det['name']} ({det['distance']}) {det['confianza']:.0f}%"
            cv2.putText(output, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

        for (ux, uy, uw, uh) in unknown_boxes:
            rx = int(ux * scale_x)
            ry = int(uy * scale_y)
            rw = int(uw * scale_x)
            rh = int(uh * scale_y)

            cv2.rectangle(output, (rx, ry), (rx + rw, ry + rh), (0, 140, 255), 2)
            cv2.putText(output, "DESCONOCIDO", (rx, max(ry - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 2)
    # FPS
    fps_counter += 1
    elapsed = time.perf_counter() - fps_start
    if elapsed >= 1.0:
        fps = fps_counter / elapsed
        fps_counter = 0
        fps_start = time.perf_counter()

    cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("DETECCION ORB + DESCONOCIDOS", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 