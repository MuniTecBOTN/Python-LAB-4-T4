import cv2
import numpy as np
import os
import math

# ==========================================
# CONFIGURACIÓN Y PARÁMETROS DEL SISTEMA
# ==========================================
REF_DIR = "wdeleon/referencias"
CATEGORIES = ["pieza_A", "pieza_B", "pieza_C"]

# Umbrales de decisión
MIN_INLIERS_HOMOGRAPHY = 12       # Mínimo de inliers RANSAC para confirmar coincidencia
RATIO_TEST_THRESH = 0.75          # Relación de Lowe para emparejamiento ORB
MIN_KEYPOINTS_EVALUABLE = 15      # Mínimo de puntos clave para considerar la pieza evaluable
MIN_AREA = 1500                   # Área mínima del objeto en píxeles
MAX_AREA = 50000                  # Área máxima del objeto en píxeles
BLUR_THRESHOLD = 50.0             # Varianza Laplaciana mínima para nitidez

class ObjectTracker:
    """Clase para gestionar el seguimiento y conteo de piezas individuales"""
    def __init__(self, track_id, centroid, category):
        self.id = track_id
        self.centroid = centroid  # (x, y)
        self.category = category  # 'pieza_A', 'pieza_B', 'pieza_C', 'Desconocida'
        self.counted = False
        self.missing_frames = 0
        self.history = [centroid]

    def update(self, new_centroid):
        self.centroid = new_centroid
        self.history.append(new_centroid)
        if len(self.history) > 30:
            self.history.pop(0)
        self.missing_frames = 0

def load_reference_templates():
    """Carga y calcula descriptores ORB para las 4 imágenes de cada categoría."""
    ref_data = {}
    orb = cv2.ORB_create(nfeatures=1500)
    
    print("[INFO] Cargando plantillas de referencia desde:", REF_DIR)
    for cat in CATEGORIES:
        ref_data[cat] = []
        cat_path = os.path.join(REF_DIR, cat)
        if not os.path.exists(cat_path):
            print(f"[ADVERTENCIA] No se encontró el directorio: {cat_path}")
            continue
            
        for i in range(1, 5):
            img_path = os.path.join(cat_path, f"imagen_{i}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, des = orb.detectAndCompute(img, None)
                    if des is not None:
                        ref_data[cat].append({
                            "kp": kp,
                            "des": des,
                            "img": img
                        })
                        print(f"  -> Cargada: {cat}/imagen_{i}.jpg ({len(kp)} Puntos clave)")
    return ref_data

def evaluate_and_match(roi_gray, ref_data, orb, bf):
    """
    Evalúa calidad, extrae ORB y compara contra plantillas usando Homografía y RANSAC.
    Retorna: (categoria, color_borde, mensaje_estado)
    """
    # 1. Validación de nitidez (Enfoque / Blur)
    laplacian_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return "No evaluable", (0, 255, 255), "No evaluable (Baja nitidez)"

    # 2. Extracción ORB en el objeto detectado
    kp_obj, des_obj = orb.detectAndCompute(roi_gray, None)
    
    if des_obj is None or len(kp_obj) < MIN_KEYPOINTS_EVALUABLE:
        return "No evaluable", (0, 255, 255), "No evaluable (Pocos detalles)"

    best_inliers = 0
    best_category = "Desconocida"

    # 3. Comparación con cada categoría y sus 4 vistas de referencia
    for cat, templates in ref_data.items():
        for tpl in templates:
            matches = bf.knnMatch(des_obj, tpl["des"], k=2)
            
            # Aplicar Test de Lowe
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < RATIO_TEST_THRESH * n.distance:
                        good_matches.append(m)

            if len(good_matches) > 8:
                src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([tpl["kp"][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Verificación geométrica con RANSAC y Homografía
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(np.sum(mask))
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_category = cat

    # 4. Decisión final según umbrales
    if best_inliers >= MIN_INLIERS_HOMOGRAPHY:
        colors = {
            "pieza_A": (0, 255, 0),      # Verde
            "pieza_B": (255, 100, 0),    # Azul / Celeste
            "pieza_C": (0, 165, 255)     # Naranja
        }
        return best_category, colors.get(best_category, (0, 255, 0)), f"{best_category} ({best_inliers} inliers)"
    else:
        # Evaluación válida pero sin coincidencia con ninguna plantilla -> Desconocida
        return "Desconocida", (0, 0, 255), f"¡DESCONOCIDA! (Inliers: {best_inliers})"

def main():
    # Inicializar captura de video (0 para cámara web por defecto o ruta de video)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara o fuente de video.")
        return

    ref_data = load_reference_templates()
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Contadores de producción
    counters = {
        "pieza_A": 0,
        "pieza_B": 0,
        "pieza_C": 0,
        "Desconocida": 0
    }

    next_object_id = 0
    tracked_objects = {}

    # Configuración de la línea virtual de conteo (se definirá dinámicamente con el alto del frame)
    line_y_pos = None

    print("\n[INFO] Sistema de inspección iniciado. Presione 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ADVERTENCIA] No se pudo recibir frame de la cámara. Saliendo...")
            break

        h, w, _ = frame.shape
        if line_y_pos is None:
            line_y_pos = int(h * 0.65) # Línea virtual al 65% de la altura de la banda

        # Preprocesamiento para segmentación en banda transportadora
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binarización adaptativa o umbral para aislar objetos sobre la banda
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Operaciones morfológicas para limpiar ruido
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_frame_centroids = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx, cy = x + bw // 2, y + bh // 2
                current_frame_centroids.append((cx, cy, x, y, bw, bh, cnt))

        # Asociación y seguimiento de objetos (Matching con IDs existentes o creación de nuevos)
        updated_tracked_objects = {}
        
        for (cx, cy, x, y, bw, bh, cnt) in current_frame_centroids:
            matched_id = None
            min_dist = 50  # Distancia máxima en píxeles para asociar el mismo objeto
            
            for obj_id, obj in tracked_objects.items():
                dist = math.hypot(obj.centroid[0] - cx, obj.centroid[1] - cy)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id

            # Extraer ROI de la pieza detectada
            roi_gray = gray[y:y+bh, x:x+bw]
            
            if roi_gray.size == 0:
                continue

            # Evaluar y clasificar la pieza
            category, color, status_text = evaluate_and_match(roi_gray, ref_data, orb, bf)

            if matched_id is not None:
                # Actualizar objeto existente
                obj = tracked_objects[matched_id]
                obj.update((cx, cy))
                obj.category = category
                obj_id = matched_id
            else:
                # Crear nuevo objeto rastreador
                obj_id = next_object_id
                next_object_id += 1
                obj = ObjectTracker(obj_id, (cx, cy), category)

            # Lógica de conteo por línea virtual
            # Si cruza la línea virtual de arriba hacia abajo y no ha sido contado
            if obj.history[0][1] < line_y_pos <= cy and not obj.counted:
                if category in counters:
                    counters[category] += 1
                obj.counted = True

            updated_tracked_objects[obj_id] = obj

            # Dibujar elementos visuales en pantalla
            cv2.drawContours(frame, [cnt], -1, color, 2)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            
            # Texto descriptivo y alerta visual si es desconocida
            label = f"ID:{obj_id} | {status_text}"
            cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if category == "Desconocida":
                cv2.putText(frame, "ALERTA: PIEZA DESCONOCIDA", (x, y + bh + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        tracked_objects = updated_tracked_objects

        # Dibujar la línea virtual de conteo
        cv2.line(frame, (0, line_y_pos), (w, line_y_pos), (0, 255, 255), 2)
        cv2.putText(frame, "Línea de Conteo Virtual", (10, line_y_pos - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Mostrar Panel de Contadores en pantalla
        panel_y = 30
        cv2.putText(frame, "--- CONTADOR DE PRODUCCIÓN ---", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for idx, (cat, count) in enumerate(counters.items()):
            text = f"{cat}: {count}"
            cv2.putText(frame, text, (10, panel_y + 25 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if cat != "Desconocida" else (0, 0, 255), 2)

        # Mostrar ventana con el resultado
        cv2.imshow("Sistema de Inspección de Piezas - WDeleon", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()