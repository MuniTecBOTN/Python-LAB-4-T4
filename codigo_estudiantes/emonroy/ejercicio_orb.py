import cv2
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN DEL SISTEMA INDUSTRIAL (12 IMÁGENES)
# ==========================================
# El sistema cargará automáticamente desde imagen_1.jpg hasta imagen_12.jpg
TOTAL_PLANTILLAS = 12
FORMATO_NOMBRES = "imagen_{}.jpg"
ID_CAMARA = 0  # Ajustar ID de la cámara conectada

def preprocesamiento_antidesenfoque_banda(imagen):
    """
    Mantiene la resolución completa de la cámara, aplicando un umbralizado
    adaptativo local básico para separar el fondo blanco de las siluetas.
    """
    if imagen is None:
        return None, []
    
    h_frame, w_frame = imagen.shape[:2]
    
    # 1. Escala de grises directa sin filtros complejos pesados
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    difuminado = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # 2. Umbralización adaptativa estándar para eliminar fondo blanco y sombras
    mascara_adaptativa = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 51, 7
    )
    
    kernel_morfo = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara_limpia = cv2.morphologyEx(mascara_adaptativa, cv2.MORPH_OPEN, kernel_morfo)
    mascara_perfecta = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel_morfo)
    
    # 3. Extraer contornos de los objetos
    contornos, _ = cv2.findContours(mascara_perfecta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    siluetas_fisicas = []
    MARGEN = 25  # Zona de exclusión para evitar cuadros rojos en las esquinas del marco
    
    for c in contornos:
        if cv2.contourArea(c) > 2000:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Descartar ruidos que toquen los bordes extremos del lente
            if (bx > MARGEN) and (by > MARGEN) and ((bx + bw) < (w_frame - MARGEN)) and ((by + bh) < (h_frame - MARGEN)):
                siluetas_fisicas.append((bx, by, bw, bh))
    
    return gris, siluetas_fisicas

def cargar_descriptores_plantillas():
    """Carga los 12 moldes configurando ORB en su nivel de análisis más bajo y básico."""
    # CONFIGURACIÓN MÍNIMA: Solo 400 puntos y una pirámide casi plana de 2 niveles
    orb = cv2.ORB_create(nfeatures=400, scaleFactor=1.2, nlevels=2)
    datos_plantillas = []
    
    print(f"--- FASE 1: CARGANDO {TOTAL_PLANTILLAS} MOLDES EN CONFIGURACIÓN MÍNIMA ---")
    for i in range(1, TOTAL_PLANTILLAS + 1):
        path = FORMATO_NOMBRES.format(i)
        if not os.path.exists(path):
            print(f"Advertencia: No se encuentra el archivo obligatorio: {path}")
            continue
        img = cv2.imread(path)
        gris_textura, _ = preprocesamiento_antidesenfoque_banda(img)
        
        kp, des = orb.detectAndCompute(gris_textura, None)
        if des is not None:
            datos_plantillas.append({
                'path': path, 
                'kp': kp, 
                'des': des, 
                'shape': gris_textura.shape, 
                'id': i
            })
            print(f"-> Firma básica para Objeto {i} ({path}) indexada con éxito.")
    return datos_plantillas

def procesar_tiempo_real():
    plantillas = cargar_descriptores_plantillas()
    if not plantillas:
        print("Error: No se pudieron cargar las plantillas de base.")
        return

    # Detector en tiempo real configurado en su nivel de exigencia más bajo y veloz
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    cap = cv2.VideoCapture(ID_CAMARA)
    if not cap.isOpened():
        print(f"Error: No se puede acceder a la cámara.")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)

    # Definición de colores BGR solicitados
    COLOR_VERDE = (0, 255, 0)   # Piezas individuales
    COLOR_AZUL = (255, 0, 0)    # Piezas juntas (Conjunto)
    COLOR_ROJO = (0, 0, 255)    # Desconocido
    
    print("\n--- ¡SISTEMA DE ANÁLISIS BÁSICO EN VIVO INICIADO! ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_visualizacion = frame.copy()
        
        # Procesar la escena manteniendo el tamaño completo original
        gray_actual, siluetas_fisicas = preprocesamiento_antidesenfoque_banda(frame)
        
        # Extraer una cantidad mínima y básica de características de la escena
        kp_escena, des_escena = orb.detectAndCompute(gray_actual, None)
        
        piezas_detectadas_este_frame = []
        siluetas_reconocidas_indices = set()

        if des_escena is not None and len(des_escena) >= 2:
            for p in plantillas:
                matches = bf.knnMatch(p['des'], des_escena, k=2)
                
                buenos_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # Ratio permisivo para emparejamientos macroscópicos simples
                        if m.distance < 0.76 * n.distance:
                            buenos_matches.append(m)

                # CONFIGURACIÓN BAJA: Solo se requieren 6 puntos coincidentes básicos para aprobar el objeto
                if len(buenos_matches) >= 6:
                    src_pts = np.float32([p['kp'][m.queryIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_escena[m.trainIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)
                    
                    M, mascara = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        mask_flat = mascara.ravel()
                        puntos_validos = dst_pts[mask_flat == 1]
                        
                        # Validación geométrica mínima reducida
                        if len(puntos_validos) >= 4:
                            h_p, w_p = p['shape']
                            lista_coordenadas = [(0, 0), (0, h_p - 1), (w_p - 1, h_p - 1), (w_p - 1, 0)]
                            esquinas_p = np.float32(lista_coordenadas).reshape(-1, 1, 2)
                            
                            esquinas_dst = cv2.perspectiveTransform(esquinas_p, M)
                            
                            centro_x = int(np.mean(puntos_validos[:, 0, 0]))
                            centro_y = int(np.mean(puntos_validos[:, 0, 1]))
                            
                            piezas_detectadas_este_frame.append({
                                'id': p['id'],
                                'centro': (centro_x, centro_y),
                                'esquinas': esquinas_dst
                            })
                            
                            for s_idx, (bx, by, bw, bh) in enumerate(siluetas_fisicas):
                                if bx <= centro_x <= (bx + bw) and by <= centro_y <= (by + bh):
                                    siluetas_reconocidas_indices.add(s_idx)

        # 3. LÓGICA DE CONTROL DE COLOR SINCRO-PERFECTA
        total_reconocidas = len(piezas_detectadas_este_frame)
        color_actual_validas = COLOR_AZUL if total_reconocidas >= 2 else COLOR_VERDE

        # Dibujar los recuadros sobre el tamaño de video completo original
        for pieza in piezas_detectadas_este_frame:
            cx, cy = pieza['centro']
            cv2.polylines(frame_visualizacion, [np.int32(pieza['esquinas'])], True, color_actual_validas, 2, cv2.LINE_AA)
            cv2.putText(frame_visualizacion, f"Molde Imagen {pieza['id']} OK", (cx - 55, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_actual_validas, 2)

        # 4. Alerta ROJA inmediata si una silueta física avanza sin emparejamiento básico aprobado
        for s_idx, (bx, by, bw, bh) in enumerate(siluetas_fisicas):
            if s_idx not in siluetas_reconocidas_indices:
                cv2.rectangle(frame_visualizacion, (bx, by), (bx + bw, by + bh), COLOR_ROJO, 2)
                cv2.putText(frame_visualizacion, "PIEZA NO RECONOCIDA", (bx, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROJO, 2)

        # Telemetría en pantalla completa
        cv2.putText(frame_visualizacion, f"Banda RealTime | Registradas: {total_reconocidas} | Desconocidas: {len(siluetas_fisicas) - len(siluetas_reconocidas_indices)}", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Sistema Industrial De Banda Transportadora - Tiempo Real", frame_visualizacion)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    procesar_tiempo_real()











