
import cv2
import numpy as np
import os

# FASE 1: CARGAR Y VALIDAR LA IMAGEN
carpeta = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(carpeta, "japan.jpg")

img = cv2.imread(ruta)

if img is None:
    print("Error: No se pudo cargar foto.jpg")
    exit()

print("Imagen cargada correctamente")

# FASE 2: PROPIEDADES
alto, ancho, canales = img.shape

print("Ancho:", ancho)
print("Alto:", alto)
print("Canales:", canales)
print("Total de pixeles:", alto * ancho)
print("Tipo de datos:", img.dtype)

# FASE 3: ESCALA DE GRISES
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FASE 4: AUMENTAR BRILLO
brillo = cv2.convertScaleAbs(
    img, alpha=1.0, beta=40
)

# FASE 5: AUMENTAR CONTRASTE
contraste = np.clip(
    img.astype(np.float32) * 1.5,
    0, 255
).astype(np.uint8)

# FASE 6: CALCULAR HISTOGRAMA
hist = cv2.calcHist(
    [gris], [0], None, [256], [0, 256]
)

# Dibujar el histograma
hist_normalizado = cv2.normalize(
    hist, None, 0, 280, cv2.NORM_MINMAX
)

grafica = np.zeros(
    (300, 512, 3), dtype=np.uint8
)

for x in range(256):
    altura = int(hist_normalizado[x])
    cv2.line(
        grafica,
        (x * 2, 299),
        (x * 2, 299 - altura),
        (0, 255, 0),
        1
    )

# FASE 7: REDIMENSIONAMIENTO
reducida = cv2.resize(
    img, (320, 240),
    interpolation=cv2.INTER_AREA
)

ampliada = cv2.resize(
    img, None,
    fx=1.5, fy=1.5,
    interpolation=cv2.INTER_CUBIC
)

# FASE 8: RECORTE
# Seleccionamos la parte central
y1, y2 = alto // 4, 3 * alto // 4
x1, x2 = ancho // 4, 3 * ancho // 4

recorte = img[y1:y2, x1:x2]

# FASE 9: GUARDAR RESULTADOS
salida = os.path.join(carpeta, "resultados")
os.makedirs(salida, exist_ok=True)

resultados = {
    "gris.png": gris,
    "brillo.png": brillo,
    "contraste.png": contraste,
    "histograma.png": grafica,
    "reducida.png": reducida,
    "ampliada.png": ampliada
}

if recorte.size > 0:
    resultados["recorte.png"] = recorte

for nombre, imagen in resultados.items():
    cv2.imwrite(
        os.path.join(salida, nombre),
        imagen
    )

print("Resultados guardados correctamente")

# MOSTRAR ALGUNOS RESULTADOS
cv2.imshow("Original", img)
cv2.imshow("Escala de grises", gris)
cv2.imshow("Histograma", grafica)

cv2.waitKey(0)
cv2.destroyAllWindows()
