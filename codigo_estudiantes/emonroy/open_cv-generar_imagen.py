import numpy as np
import cv2 as cv


imagen = np.array([[0,  50,  100], 
                   [150,  200,  255],
                   [255,  100,  0]
                   ],dtype=np.uint8)


print(imagen.shape)
print(imagen[0, 0])
print(imagen[1, 2])


imagen_grande = cv.resize(imagen, (300, 300), interpolation=cv.INTER_NEAREST)

cv.imshow ("Imagen ampliada", imagen_grande)

# mostrar la imagen en una ventana
cv.imshow ("mi primera imagen", imagen)

#esperar hasta que se presine una tecla
cv.waitKey(0)

# cerrar las ventanas
cv.destroyAllWindows()