import numpy as np
import cv2 as cv

imagen = np.array([
    [255, 255, 255],
    [255, 0, 255],
    [255, 255, 255]
    
], dtype=np.uint8)

print(imagen.shape)
print(imagen[0,0])
print(imagen[1,2])

imagen_grande = cv.resize(imagen, (300, 300), interpolation=cv.INTER_CUBIC)
cv.imshow("Imagen Grande", imagen_grande)
cv.waitKey(0)
cv.destroyAllWindows()
