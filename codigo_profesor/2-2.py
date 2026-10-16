
import cv2

imagen = cv2.imread("objetos.jpg")

if imagen is None:
    raise FileNotFoundError("No se encontró objetos.jpg")

# Rectángulo: punto inicial, punto final, color BGR y grosor
cv2.rectangle(
    imagen, (50, 50), (250, 200),
    (0, 255, 0), 3
)

# Círculo: centro, radio, color y grosor
cv2.circle(
    imagen, (350, 150), 50,
    (255, 0, 0), 3
)

# Texto: contenido, posición, fuente, tamaño y color
cv2.putText(
    imagen, "Objeto detectado", (50, 35),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
    (0, 0, 255), 2
)

cv2.imshow("Imagen anotada", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
