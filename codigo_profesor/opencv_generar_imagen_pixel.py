import numpy as np
import cv2 as cv

imagen = np.array(
    [
        [[2,33,255], [255, 255, 255], [255, 255, 255]],
        [[255, 0, 255], [255, 255, 255], [255, 255, 255]],
        [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
    ],
    dtype=np.uint8,
)

print(imagen.shape)  # (3, 3)
print(imagen[0, 0])  # 0 → esquina sup. izq.
print(imagen[1, 2])  # 255 → fila 1, col. 2

# Ampliar la imagen 100 veces
imagen_grande = cv.resize(imagen, (300, 300), interpolation=cv.INTER_AREA)

cv.imshow("Imagen ampliada", imagen_grande)

# Esperar hasta que se presione una tecla
cv.waitKey(0)

# Cerrar las ventanas
cv.destroyAllWindows()
