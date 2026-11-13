import cv2
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN DEL SISTEMA INDUSTRIAL (12 IMÁGENES)
# ==========================================
TOTAL_PLANTILLAS = 12
FORMATO_NOMBRES = "imagen_{}.jpg"
ID_CAMARA = 1  # Ajustar ID de la cámara conectada
TAMANO_ESTANDAR = (300, 300)  # Normaliza escala para que funcione de cerca y de lejos
MIN_COINCIDENCIAS = 15        # Mínimo de puntos buenos para confirmar un objeto

# Inicializamos un único detector ORB robusto y el comparador
orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def mejorar_textura_orb(imagen_gris):
    """
    Limpia ruido de compresión JPG y equilibra la iluminación local
    usando CLAHE para que ORB funcione bajo cualquier luz.
    """
    limpia = cv2.bilateralFilter(imagen_gris, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(limpia)


def detectar_objetos_en_escena(frame):
    """
    Encuentra las cajas delimitadoras (ROI) de los objetos sobre el fondo.
    """
    h_frame, w_frame = frame.shape[:2]
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (5, 5), 0)

    mascara = cv2.adaptiveThreshold(
        suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 7
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas_validas = []
    MARGEN = 20

    for c in contornos:
        if cv2.contourArea(c) > 2500:
            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx > MARGEN) and (by > MARGEN) and ((bx + bw) < (w_frame - MARGEN)) and ((by + bh) < (h_frame - MARGEN)):
                cajas_validas.append((bx, by, bw, bh))

    return gris, cajas_validas


def cargar_descriptores_plantillas():
    """
    Carga las 12 imágenes .jpg, recorta el objeto si tiene fondo extra,
    lo normaliza a 300x300 y extrae sus puntos clave.
    """
    datos_plantillas = []
    print(f"--- CARGANDO {TOTAL_PLANTILLAS} PLANTILLAS OPTIMIZADAS ---")

    for i in range(1, TOTAL_PLANTILLAS + 1):
        path = FORMATO_NOMBRES.format(i)
        if not os.path.exists(path):
            print(f"[!] Advertencia: No se encuentra {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        gris, cajas = detectar_objetos_en_escena(img)
        if len(cajas) > 0:
            bx, by, bw, bh = max(cajas, key=lambda b: b[2] * b[3])
            recorte = gris[by:by+bh, bx:bx+bw]
        else:
            recorte = gris

        recorte_std = cv2.resize(recorte, TAMANO_ESTANDAR)
        textura_lista = mejorar_textura_orb(recorte_std)

        kp, des = orb.detectAndCompute(textura_lista, None)
        if des is not None and len(kp) > 10:
            datos_plantillas.append({
                "id": i,
                "nombre": f"Objeto {i}",
                "kp": kp,
                "des": des
            })
            print(f"-> {path} indexado correctamente ({len(kp)} puntos clave).")
        else:
            print(f"[!] {path} tiene muy poca textura para ORB.")

    return datos_plantillas


def identificar_recorte(des_roi, plantillas):
    """
    Compara un recorte de la cámara contra las 12 plantillas usando
    el Filtro de Lowe (Ratio Test) y devuelve el mejor candidato.
    """
    if des_roi is None or len(des_roi) < 10:
        return None, 0

    mejor_objeto = None
    max_puntos_buenos = 0

    for plantilla in plantillas:
        matches = bf.knnMatch(plantilla["des"], des_roi, k=2)
        puntos_buenos = 0

        for par in matches:
            if len(par) == 2:
                m, n = par
                if m.distance < 0.75 * n.distance:
                    puntos_buenos += 1

        if puntos_buenos > max_puntos_buenos:
            max_puntos_buenos = puntos_buenos
            mejor_objeto = plantilla["nombre"]

    if max_puntos_buenos >= MIN_COINCIDENCIAS:
        return mejor_objeto, max_puntos_buenos

    return None, max_puntos_buenos


# ==========================================
# BUCLE PRINCIPAL EN TIEMPO REAL
# ==========================================
if __name__ == "__main__":
    plantillas = cargar_descriptores_plantillas()

    if len(plantillas) == 0:
        print("Error: No se cargó ninguna plantilla válida. Revisa tus archivos .jpg.")
        exit()

    cap = cv2.VideoCapture(ID_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n--- SISTEMA INICIADO (Presiona 'q' para salir) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer la cámara.")
            break

        gris_frame, cajas = detectar_objetos_en_escena(frame)

        for (bx, by, bw, bh) in cajas:
            # 1. Extraer solo el cuadro del objeto y normalizarlo a 300x300
            recorte_cam = gris_frame[by-50:by+bh +50, bx-50:bx+bw+50]
            recorte_std = cv2.resize(recorte_cam, TAMANO_ESTANDAR)
            textura_cam = mejorar_textura_orb(recorte_std)
            #cv2.imshow("descriptores", textura_cam)
            # 2. Extraer puntos clave del recorte
            _, des_cam = orb.detectAndCompute(textura_cam, None)

            # 3. Comparar con las 12 plantillas
            nombre_detectado, aciertos = identificar_recorte(des_cam, plantillas)

            # 4. Interfaz visual según el resultado
            if nombre_detectado is not None:
                # VERDE: Objeto reconocido
                color = (0, 255, 0)
                etiqueta = f"Objeto reconocido: {nombre_detectado}"
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 3)
                cv2.rectangle(frame, (bx, by - 30), (bx + max(bw, 260), by), color, -1)
                cv2.putText(frame, etiqueta, (bx + 5, by - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            else:
                # ROJO: Objeto no reconocido (BGR: Azul=0, Verde=0, Rojo=255)
                color = (0, 0, 255)
                etiqueta = "Objeto no reconocido"
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 3)
                cv2.rectangle(frame, (bx, by - 30), (bx + max(bw, 210), by), color, -1)
                cv2.putText(frame, etiqueta, (bx + 5, by - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Reconocimiento Industrial ORB - Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()













