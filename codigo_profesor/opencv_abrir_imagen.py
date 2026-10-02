import cv2  # Importa la biblioteca OpenCV

imagen = cv2.imread("imagen.jpg")  # Carga la imagen desde un archivo

imagen = cv2.resize(imagen, (800, 600))  # Cambia el tamaño a 800 píxeles de ancho y 600 de alto

cv2.imshow("Mi imagen", imagen)  # Muestra la imagen en una ventana llamada "Mi imagen"

cv2.waitKey(0)  # Espera hasta que el usuario presione una tecla

cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas por OpenCV