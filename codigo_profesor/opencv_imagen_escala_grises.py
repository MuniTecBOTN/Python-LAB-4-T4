import cv2  # Importa la biblioteca OpenCV

imagen = cv2.imread("imagen.jpg")  # Carga una imagen desde un archivo

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises

cv2.imshow("Imagen original", imagen)  # Muestra la imagen original a color

cv2.imshow("Imagen en gris", gris)  # Muestra la imagen convertida a escala de grises

cv2.waitKey(0)  # Espera hasta que el usuario presione una tecla

cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas