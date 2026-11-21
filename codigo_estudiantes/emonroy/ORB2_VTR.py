import cv2

# 1. Inicializar la cámara (0 es el índice de la cámara integrada)
cap = cv2.VideoCapture(0)

# 2. Crear el objeto ORB
orb = cv2.ORB_create(nfeatures=500)

print("Presiona la tecla 'q' para salir.")

while True:
    # 3. Capturar fotograma por fotograma
    ret, frame = cap.read()
    
    # Si la cámara no responde correctamente, salir del bucle
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # 4. Convertir el fotograma a escala de grises (necesario para ORB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Detectar puntos clave y calcular descriptores
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 6. Dibujar los puntos clave sobre el fotograma original a color
    # Usamos flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS para ver el tamaño y orientación
    img_keypoints = cv2.drawKeypoints(
        frame, keypoints, None, color=(0, 255, 0), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 7. Mostrar el fotograma resultante en una ventana
    cv2.imshow('Deteccion ORB en Tiempo Real', img_keypoints)

    # 8. Detener el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Liberar la cámara y cerrar las ventanas al terminar
cap.release()
cv2.destroyAllWindows()
