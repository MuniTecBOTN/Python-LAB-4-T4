import cv2

# 1. Inicializar la cámara
cap = cv2.VideoCapture(0)

# 2. Crear el objeto ORB
orb = cv2.ORB_create(nfeatures=1500)

print("Controles:")
print(" - Presiona 's' o ESPACIO para tomar una FOTO.")
print(" - Presiona 'q' para SALIR.")

# Contador para nombrar las fotos de forma consecutiva
contador_fotos = 1

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Clonamos el frame original antes de dibujarle encima para poder guardar la foto limpia
    foto_limpia = frame.copy()

    # Convertir a escala de grises y calcular ORB para la visualización en pantalla
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Dibujar los puntos clave en el frame que se muestra en vivo
    img_keypoints = cv2.drawKeypoints(
        frame, keypoints, None, color=(0, 255, 0), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Mostrar el video con los ORB en tiempo real
    cv2.imshow('Deteccion ORB en Tiempo Real', img_keypoints)

    # Capturar la tecla presionada
    tecla = cv2.waitKey(1) & 0xFF

    # --- LÓGICA PARA TOMAR LA FOTO ---
    # Si presionas la tecla 's' o la barra espaciadora (código 32)
    if tecla == ord('s') or tecla == 32:
        nombre_archivo = f"captura_{contador_fotos}.jpg"
        
        # cv2.imwrite guarda la imagen en tu disco duro
        cv2.imwrite(nombre_archivo, foto_limpia)
        
        print(f"¡Foto guardada con éxito como: {nombre_archivo}!")
        contador_fotos += 1 # Suma 1 para que la siguiente foto no reescriba esta

    # --- LÓGICA PARA SALIR ---
    elif tecla == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado correctamente.")
