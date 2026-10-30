import cv2
import numpy as np

# 1. Cargar la imagen usando la ruta absoluta exacta de Windows
imagen = cv2.imread('manzanas.jpg')

# VALIDACIÓN: Verifica si OpenCV realmente pudo cargar el archivo
if imagen is None:
 print("Error: No se pudo cargar la imagen. Verifica que el nombre o la extensión (.jpg, .png) sean correctos.")
else:
 # Si la imagen existe, continúa con la conversión
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 2. Aplicar un filtro de desfoque para reducir variaciones de luz
desfoque = cv2.GaussianBlur(gris, (5, 5), 0)

# 3. Binarizar la imagen (Convertir a blanco y negro puro)
# El fondo debe ser negro y los objetos blancos para findContours
_, binarizada = cv2.threshold(desfoque, 127, 255, cv2.THRESH_BINARY_INV)

# 4. Encontrar los contornos de los objetos
contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copia de la imagen original para dibujar los resultados
resultado = imagen.copy()

print(f"Se encontraron un total de {len(contornos)} contornos candidatos.\n")

# Establecer un umbral de área mínima para filtrar el ruido (en píxeles)
AREA_MINIMA = 500

# 5. Iterar sobre cada contorno para calcular su área
for i, contorno in enumerate(contornos):
     # CALCULAR EL ÁREA EN PÍXELES
     area = cv2.contourArea(contorno)

 # Filtrar objetos que son demasiado pequeños (ruido)
if area > AREA_MINIMA:
    print(f"Contorno #{i}: Área válida = {area:.2f} píxeles")

 # Dibujar el contorno en color verde (grosor de 2 píxeles)
    cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 2)
 
 # Opcional: Obtener coordenadas para poner texto encima del objeto
    x, y, w, h = cv2.boundingRect(contorno)
    cv2.putText(resultado, f"Area: {int(area)}", (x, y - 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
else:
    print(f"Contorno #{i}: Ignorado por ruido (Área = {area:.2f} píxeles)")

# 6. Mostrar las imágenes en ventanas
    cv2.imshow("Imagen Binaria (Entrada)", binarizada)
    cv2.imshow("Objetos Detectados (Filtrados)", resultado)

# Esperar a que se presione una tecla para cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()
