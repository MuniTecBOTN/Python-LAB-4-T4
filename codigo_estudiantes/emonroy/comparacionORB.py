import cv2
import numpy as np

# 1. Cargar la imagen de referencia (.jpg) en ambas versiones
imagen_ref_color = cv2.imread('captura_4.jpg', cv2.IMREAD_COLOR)
imagen_ref_gris = cv2.imread('captura_4.jpg', cv2.IMREAD_GRAYSCALE)

if imagen_ref_color is None or imagen_ref_gris is None:
    print("Error: No se pudo cargar la imagen de referencia. Verifica el nombre del archivo.")
    exit()

# === MEJORA DE FONDO: Máscara binaria para ignorar el fondo negro y sus matices ===
_, mask_ref = cv2.threshold(imagen_ref_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imagen_ref_gris_clean = cv2.bitwise_and(imagen_ref_gris, imagen_ref_gris, mask=mask_ref)

h_ref, w_ref = imagen_ref_gris.shape

# 2. Inicializar la cámara
cap = cv2.VideoCapture(0)

# 3. Configuración de ORB para enfocarse en contornos claros
orb = cv2.ORB_create(
    nfeatures=1500,        
    scaleFactor=1.1,       
    nlevels=8,            
    scoreType=cv2.ORB_HARRIS_SCORE,
    edgeThreshold=15       
)

# Calcular descriptores únicamente en la zona limpia del objeto
kp_ref, des_ref = orb.detectAndCompute(imagen_ref_gris_clean, None)

# 4. Buscador FLANN
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Variables de Interfaz
modo_color = True
ver_lineas = True
sensibilidad = 0.80  
nombre_ventana = 'Suite de Analisis Definitivo ORB'

print("\n=== CONTROLES DE LA SUITE DE ANÁLISIS ===")
print("[C] - Alternar entre imagen a COLOR / ESCALA DE GRISES")
print("[L] - Activar / Desactivar LÍNEAS de unión")
print("[MAS +] - Dar más tolerancia a distancias largas (+ emparejamientos)")
print("[MENOS -] - Reducir la tolerancia (Análisis geométrico más estricto)")
print("[Q] - Salir\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el fotograma de la cámara.")
        break

    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === SEGMENTACIÓN EN TIEMPO REAL ===
    blurred_frame = cv2.GaussianBlur(frame_gris, (5, 5), 0)
    frame_thresh = cv2.adaptiveThreshold(
        blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    frame_gris_clean = cv2.bitwise_and(frame_gris, frame_gris, mask=frame_thresh)

    kp_cam, des_cam = orb.detectAndCompute(frame_gris_clean, None)

    matchesMask = None
    objeto_detectado = False
    puntos_reales = 0

    if des_ref is not None and des_cam is not None and len(des_cam) > 1:
        matches = flann.knnMatch(des_ref, des_cam, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < sensibilidad * n.distance:
                    good_matches.append(m)

        # Análisis geométrico estricto
        if len(good_matches) >= 10:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_cam[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # === VALIDACIÓN GEOMÉTRICA DE LA MATRIZ DE HOMOGRAFÍA ===
                # Calculamos el determinante de la submatriz de rotación/escala (2x2 superior izquierda)
                det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
                
                # Una homografía real y coherente debe tener un determinante positivo y no colapsado
                # Si el objeto no está, RANSAC deforma la matriz dando valores negativos o absurdos
                if det > 0.1:
                    matchesMask = mask.ravel().tolist()
                    puntos_reales = matchesMask.count(1)

                    # Subimos levemente el filtro: requerimos que al menos 12 puntos pasen la geometría estricta
                    if puntos_reales >= 12:
                        objeto_detectado = True
                        
                        # Dibujar contorno exterior
                        pts_esquinas = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
                        dst_esquinas = cv2.perspectiveTransform(pts_esquinas, M)
                        frame = cv2.polylines(frame, [np.int32(dst_esquinas)], True, (0, 255, 0), 3, cv2.LINE_AA)

    imagen_salida = frame if modo_color else cv2.cvtColor(frame_gris, cv2.COLOR_GRAYBGR)

    if ver_lineas and matchesMask is not None and objeto_detectado:
        imagen_final = cv2.drawMatches(
            imagen_ref_color if modo_color else cv2.cvtColor(imagen_ref_gris_clean, cv2.COLOR_GRAYBGR),
            kp_ref,
            imagen_salida,
            kp_cam,
            good_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    else:
        img_ref_visual = imagen_ref_color if modo_color else cv2.cvtColor(imagen_ref_gris_clean, cv2.COLOR_GRAYBGR)
        img_ref_rescaled = cv2.resize(img_ref_visual, (w_ref, frame.shape[0]))
        imagen_final = np.hstack((img_ref_rescaled, imagen_salida))

    if objeto_detectado:
        texto_estado = f"CUADRADO DETECTADO - Puntos: {puntos_reales}"
        color_interfaz = (0, 255, 0)
    else:
        texto_estado = f"BUSCANDO CUADRADO BLANCO... Sens: {sensibilidad:.2f}"
        color_interfaz = (0, 0, 255)

    cv2.putText(imagen_final, texto_estado, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_interfaz, 2, cv2.LINE_AA)
    cv2.imshow(nombre_ventana, imagen_final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('c') or key == ord('C'):
        modo_color = not modo_color
    elif key == ord('l') or key == ord('L'):
        ver_lineas = not ver_lineas
    elif key == ord('+') or key == ord('='):
        sensibilidad = min(0.98, sensibilidad + 0.02)
    elif key == ord('-') or key == ord('_'):
        sensibilidad = max(0.50, sensibilidad - 0.02)

    if cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Suite terminada.")

