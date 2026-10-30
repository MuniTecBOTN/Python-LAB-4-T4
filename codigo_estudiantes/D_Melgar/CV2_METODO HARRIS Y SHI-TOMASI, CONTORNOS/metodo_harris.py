import cv2
import numpy as np

# 1. Cargar la imagen original en color (para poder pintar las esquinas en rojo)
img = cv2.imread('Figuras.jpg')

# 2. Crear una versión en escala de grises para el algoritmo de Harris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 3. Aplicar Harris
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 4. Dilatar para marcar mejor las esquinas
dst = cv2.dilate(dst, None)

# 5. Umbral para marcar las esquinas óptimas sobre la imagen en color
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Opcional: Mostrar y guardar el resultado
cv2.imshow('Esquinas Harris', img)
cv2.waitKey(0)
cv2.destroyAllWindows()