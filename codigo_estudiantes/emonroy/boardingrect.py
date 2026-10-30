import cv2

# 1. Cargar la imagen en escala de grises (es más fácil detectar formas así)
imagen = cv2.imread('3_cuadrados.bmp')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 2. Aplicar un umbral (Binarización) para separar el objeto del fondo
# Esto convierte la imagen a blanco y negro puro
_, umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

# 3. Encontrar los contornos del objeto en la imagen binarizada
contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Iterar sobre los contornos encontrados
for contorno in contornos:
 # Ignorar contornos muy pequeños que puedan ser ruido de fondo
        if cv2.contourArea(contorno) > 100:
        
        # --- AQUÍ SE CALCULA EL BOUNDING RECT ---
        # Devuelve: x, y (coordenadas), w (ancho), h (alto)
         x, y, w, h = cv2.boundingRect(contorno)
        
        # # 5. Dibujar el rectángulo delimitador sobre la imagen original
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

 
 # Opcional: Imprimir las dimensiones en la consola
print(f"Objeto detectado en X:{x}, Y:{y} | Ancho: {w}px, Alto: {h}px")

# 6. Mostrar el resultado en una ventana
cv2.imshow('BoundingRect de OpenCV', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

