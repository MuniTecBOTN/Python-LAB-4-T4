import cv2
import matplotlib.pyplot as plt

# 1. Cargar la imagen en escala de grises
img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Crear el objeto ORB
orb = cv2.ORB_create(nfeatures=500)

# 3. Detectar puntos clave y calcular descriptores
keypoints, descriptors = orb.detectAndCompute(img, None)

# 4. Dibujar los puntos clave en la imagen
img_keypoints = cv2.drawKeypoints(
    img, keypoints, None, color=(0, 255, 0), flags=0
)

# 5. Mostrar resultado
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
