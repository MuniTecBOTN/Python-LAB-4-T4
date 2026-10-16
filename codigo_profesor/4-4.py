
import cv2

imagen = cv2.imread("edificio.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró edificio.jpg")

# Convertir a escala de grises
gris = cv2.cvtColor(
    imagen, cv2.COLOR_BGR2GRAY
)

# Reducir ruido mediante desenfoque
desenfocada = cv2.GaussianBlur(
    gris, (5, 5), 0
)

# Detectar bordes
bordes = cv2.Canny(
    desenfocada, 50, 150
)

cv2.imshow("Original", imagen)
cv2.imshow("Escala de grises", gris)
cv2.imshow("Desenfocada", desenfocada)
cv2.imshow("Bordes", bordes)

cv2.waitKey(0)
cv2.destroyAllWindows()
