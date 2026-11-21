import cv2

# 1. Cargar la imagen de referencia (la foto fija .jpg) en escala de grises
# ¡Asegúrate de cambiar 'captura_1.jpg' por el nombre real de tu foto!
imagen_referencia = cv2.imread('captura_2.jpg', cv2.IMREAD_GRAYSCALE)

if imagen_referencia is None:
    print("Error: No se pudo cargar la imagen de referencia. Verifica el nombre del archivo.")
    exit()

# 2. Inicializar la cámara
cap = cv2.VideoCapture(0)

# 3. Crear el objeto ORB
orb = cv2.ORB_create(nfeatures=1500,scaleFactor=1.2, nlevels=8)

# 4. Calcular puntos clave y descriptores para la imagen fija
kp_ref, des_ref = orb.detectAndCompute(imagen_referencia, None)

# Cantidad total de características en la imagen de referencia para usar como base del 100%
total_puntos_ref = len(kp_ref) if kp_ref is not None else 1

# 5. Crear el emparejador BFMatcher para buscar los 2 mejores vecinos (k=2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

print("Presiona la tecla 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # 6. Convertir el fotograma de la cámara a escala de grises
    gray_camara = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 7. Detectar puntos clave y descriptores en el fotograma en vivo
    kp_camara, des_camara = orb.detectAndCompute(gray_camara, None)

    # Creamos una copia del frame original para añadirle los textos después de drawMatches
    pantalla_final = frame.copy()

    # 8. Realizar la comparación mediante KNN Match
    if des_ref is not None and des_camara is not None and len(des_camara) > 1:
        # Buscamos las 2 mejores coincidencias para cada punto clave
        matches = bf.knnMatch(des_ref, des_camara, k=2)
        
        # Filtrar las mejores coincidencias usando el test de ratio de Lowe (0.75-0.80 es el estándar)
        buenas_coincidencias = []
        for m, n in matches:
            if m.distance < 0.78 * n.distance:
                buenas_coincidencias = [m] + buenas_coincidencias

        # 9. Calcular el porcentaje de coincidencia
        # Relaciona los puntos buenos encontrados con el total que tiene la imagen original
        porcentaje = (len(buenas_coincidencias) / total_puntos_ref) * 100
        if porcentaje > 100: porcentaje = 100.0  # Limitar al 100% máximo

        # 10. Dibujar la unión de puntos entre ambas imágenes
        resultado_grafico = cv2.drawMatches(
            imagen_referencia, kp_ref, 
            frame, kp_camara, 
            buenas_coincidencias[:30], None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 11. Evaluar si es el mismo objeto (Umbral sugerido: 15%)
        # Puedes subir o bajar este número según qué tan estricto quieras el filtro
        UMBRAL_MINIMO = 15.0 

        if porcentaje >= UMBRAL_MINIMO:
            texto = f"COINCIDENCIA: {porcentaje:.1f}% - MISMO OBJETO"
            color_texto = (0, 255, 0) # Verde si coinciden
        else:
            texto = "ALERTA: NO SON EL MISMO OBJETO"
            color_texto = (0, 0, 255) # Rojo si no coinciden

        # Insertar el texto decorativo directamente arriba en la ventana combinada
        cv2.putText(resultado_grafico, texto, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2, cv2.LINE_AA)
        
        cv2.imshow('Comparativo ORB en Tiempo Real', resultado_grafico)
    else:
        # Mensaje de alerta si la cámara está completamente tapada o no detecta nada
        cv2.putText(pantalla_final, "ALERTA: NO SON EL MISMO OBJETO (No detectado)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Comparativo ORB en Tiempo Real', pantalla_final)

    # 12. Detener el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Programa finalizado correctamente.")
