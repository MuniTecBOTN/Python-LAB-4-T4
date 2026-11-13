import cv2
import numpy as np
import os


TOTAL_OBJETOS = 3
FORMATO_NOMBRES = "imagen_{}.jpg"
ID_CAMARA = 0
VISTAS_POR_OBJETO = 4
TOTAL_PLANTILLAS = TOTAL_OBJETOS * VISTAS_POR_OBJETO
MIN_BUENOS_MATCHES = 12
MIN_INLIERS = 12
MIN_RATIO_INLIERS = 0.55
MARGEN_CLASIFICACION = 2.0

def preprocesamiento_antidesenfoque_banda(imagen):
    """
    Mantiene la resolución completa de la cámara, aplicando un umbralizado
    adaptativo local básico para separar el fondo blanco de las siluetas.
    """
    if imagen is None:
        return None, []
    
    h_frame, w_frame = imagen.shape[:2]
    
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    mascara_adaptativa = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 51, 7
    )
    
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_limpia = cv2.morphologyEx(mascara_adaptativa, cv2.MORPH_OPEN, kernel_apertura)
    mascara_perfecta = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel_cierre)
    
    contornos, _ = cv2.findContours(mascara_perfecta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    siluetas_fisicas = []
    MARGEN = 25
    
    for c in contornos:
        if cv2.contourArea(c) > 2000:
            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx > MARGEN) and (by > MARGEN) and ((bx + bw) < (w_frame - MARGEN)) and ((by + bh) < (h_frame - MARGEN)):
                siluetas_fisicas.append((bx, by, bw, bh))
    
    return gris, siluetas_fisicas

def cargar_descriptores_plantillas():
    """Carga las 12 vistas y las agrupa en tres objetos, cuatro vistas por objeto."""
    orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=4)
    datos_plantillas = []
    
    print(f"--- FASE 1: CARGANDO {TOTAL_PLANTILLAS} VISTAS DE {TOTAL_OBJETOS} OBJETOS ---")
    for i in range(1, TOTAL_PLANTILLAS + 1):
        path = FORMATO_NOMBRES.format(i)
        if not os.path.exists(path):
            print(f"Error: No se encuentra la vista obligatoria: {path}")
            return []
        img = cv2.imread(path)
        if img is None:
            print(f"Error: No se pudo leer la vista: {path}")
            return []
        gris_textura, _ = preprocesamiento_antidesenfoque_banda(img)
        
        kp, des = orb.detectAndCompute(gris_textura, None)
        if des is None or len(des) == 0:
            print(f"Error: No se encontraron características en la vista: {path}")
            return []
        datos_plantillas.append({
            'path': path, 
            'kp': kp, 
            'des': des, 
            'shape': gris_textura.shape, 
            'id': (i - 1) // VISTAS_POR_OBJETO + 1
        })
        print(f"-> Vista {i} indexada para Objeto {(i - 1) // VISTAS_POR_OBJETO + 1}.")
    return datos_plantillas

def clasificar_region(region, plantillas, orb, bf):
    """Devuelve el objeto mejor respaldado por coincidencias geométricas o None."""
    kp_region, des_region = orb.detectAndCompute(region, None)
    if des_region is None or len(des_region) < 2:
        return None

    mejores_por_objeto = {}
    for plantilla in plantillas:
        pares = bf.knnMatch(plantilla['des'], des_region, k=2)
        buenos_matches = []
        for par in pares:
            if len(par) == 2:
                primero, segundo = par
                if primero.distance < 0.72 * segundo.distance:
                    buenos_matches.append(primero)

        if len(buenos_matches) < MIN_BUENOS_MATCHES:
            continue

        puntos_plantilla = []
        puntos_region = []
        for match in buenos_matches:
            puntos_plantilla.append(plantilla['kp'][match.queryIdx].pt)
            puntos_region.append(kp_region[match.trainIdx].pt)

        puntos_plantilla = np.float32(puntos_plantilla).reshape(-1, 1, 2)
        puntos_region = np.float32(puntos_region).reshape(-1, 1, 2)
        _, mascara = cv2.findHomography(
            puntos_plantilla, puntos_region, cv2.RANSAC, 5.0
        )
        if mascara is None:
            continue

        inliers = int(mascara.ravel().sum())
        ratio_inliers = inliers / len(buenos_matches)
        if inliers < MIN_INLIERS or ratio_inliers < MIN_RATIO_INLIERS:
            continue

        objeto_id = plantilla['id']
        resultado = mejores_por_objeto.get(objeto_id)
        if resultado is None or inliers > resultado:
            mejores_por_objeto[objeto_id] = inliers

    mejor_id = None
    mejores_inliers = 0
    segundo_mejor = 0

    for objeto_id, inliers in mejores_por_objeto.items():
        if inliers > mejores_inliers:
            segundo_mejor = mejores_inliers
            mejores_inliers = inliers
            mejor_id = objeto_id
        elif inliers > segundo_mejor:
            segundo_mejor = inliers

    if mejor_id is None:
        return None

    if segundo_mejor > 0 and mejores_inliers < MARGEN_CLASIFICACION * segundo_mejor:
        return None
    return mejor_id

def procesar_tiempo_real():
    plantillas = cargar_descriptores_plantillas()
    if not plantillas:
        print("Error: No se pudieron cargar las plantillas de base.")
        return

    orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=4)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    cap = cv2.VideoCapture(ID_CAMARA)
    if not cap.isOpened():
        print(f"Error: No se puede acceder a la cámara.")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)

    COLOR_VERDE = (0, 255, 0)
    COLOR_AZUL = (255, 0, 0)
    COLOR_ROJO = (0, 0, 255) 
    
    print("\n--- ¡SISTEMA DE ANÁLISIS BÁSICO EN VIVO INICIADO! ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_visualizacion = frame.copy()
        
        gray_actual, siluetas_fisicas = preprocesamiento_antidesenfoque_banda(frame)
        
        piezas_detectadas_este_frame = []
        siluetas_reconocidas_indices = set()

        for s_idx, (bx, by, bw, bh) in enumerate(siluetas_fisicas):
            roi = gray_actual[by:by + bh, bx:bx + bw]
            objeto_reconocido = clasificar_region(roi, plantillas, orb, bf)
            if objeto_reconocido is not None:
                siluetas_reconocidas_indices.add(s_idx)

            piezas_detectadas_este_frame.append({
                'id': objeto_reconocido,
                'caja': (bx, by, bw, bh)
            })

        total_reconocidas = len(siluetas_reconocidas_indices)
        color_actual_validas = COLOR_AZUL if total_reconocidas >= 2 else COLOR_VERDE

        for pieza in piezas_detectadas_este_frame:
            bx, by, bw, bh = pieza['caja']
            if pieza['id'] is not None:
                cv2.rectangle(frame_visualizacion, (bx, by), (bx + bw, by + bh), color_actual_validas, 2)
                etiqueta = f"Objeto {pieza['id']} reconocido"
                color_etiqueta = color_actual_validas
            else:
                cv2.rectangle(frame_visualizacion, (bx, by), (bx + bw, by + bh), COLOR_ROJO, 2)
                etiqueta = "Objeto no reconocido"
                color_etiqueta = COLOR_ROJO
            cv2.putText(frame_visualizacion, etiqueta, (bx, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_etiqueta, 2)

        cv2.putText(frame_visualizacion, f"Objetos reconocidos: {total_reconocidas} | No reconocidos: {len(siluetas_fisicas) - len(siluetas_reconocidas_indices)}", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Sistema Industrial De Banda Transportadora - Tiempo Real", frame_visualizacion)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    procesar_tiempo_real()