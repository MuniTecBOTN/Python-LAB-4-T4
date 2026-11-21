import cv2

# 1. Cargar la imagen de referencia (la foto fija .jpg) en escala de grises
# ¡Asegúrate de cambiar 'captura_1.jpg' por el nombre real de tu foto!
imagen_referencia = cv2.imread('captura_1.jpg', cv2.IMREAD_GRAYSCALE)

if imagen_referencia is None:
    print("Error: No se pudo cargar la imagen de referencia. Verifica el nombre del archivo.")
    exit()

# 2. Inicializar la cámara
cap = cv2.VideoCapture(0)

# 3. Crear el objeto ORB
orb = cv2.ORB_create(nfeatures=500)

# 4. Calcular puntos clave y descriptores SOLO PARA LA IMAGEN FIJA (se hace una sola vez)
kp_ref, des_ref = orb.detectAndCompute(imagen_referencia, None)

# 5. Crear el formateador/emparejador de coincidencias (Brute-Force Matcher)
# Usamos NORM_HAMMING porque ORB genera descriptores binarios
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("Presiona la tecla 'q' para salir.")

while True:
    # 6. Capturar fotograma por fotograma
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # 7. Convertir el fotograma de la cámara a escala de grises
    gray_camara = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 8. Detectar puntos clave y descriptores en el fotograma en vivo
    kp_camara, des_camara = orb.detectAndCompute(gray_camara, None)

    # 9. Realizar la comparación si ambos descriptores tienen datos
    if des_ref is not None and des_camara is not None:
        # Buscar las mejores coincidencias entre la foto fija y la cámara
        coincidencias = bf.match(des_ref, des_camara)
        
        # Ordenar las coincidencias por distancia (las más cercanas/parecidas primero)
        coincidencias = sorted(coincidencias, key=lambda x: x.distance)

        # 10. Dibujar las primeras 30 mejores coincidencias en una pantalla compartida
        # Esto conectará con líneas los puntos parecidos entre ambas imágenes
        resultado_comparativo = cv2.drawMatches(
            imagen_referencia, kp_ref, 
            frame, kp_camara, 
            coincidencias[:30], None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Mostrar la ventana con la comparación lado a lado
        cv2.imshow('Comparativo ORB en Tiempo Real', resultado_comparativo)
    else:
        # Si la cámara no detecta nada, mostrar el fotograma normal
        cv2.imshow('Comparativo ORB en Tiempo Real', frame)

    # 11. Detener el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 12. Liberar la cámara y cerrar las ventanas al terminar
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado correctamente.")
