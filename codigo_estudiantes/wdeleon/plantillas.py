import cv2
import numpy as np
from pathlib import Path

class Plantilla:
    def __init__(self, nombre, ruta):
        self.nombre = nombre
        self.img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"No se pudo leer {ruta}")
        self.kp, self.des = None, None
        self._extraer_orb()

    def _extraer_orb(self):
        orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=15, fastThreshold=10)
        self.kp, self.des = orb.detectAndCompute(self.img, None)


class RegistroPlantillas:
    """Carga las 3 piezas y agrupa sus 4 vistas cada una."""
    def __init__(self, raiz="referencias"):
        self.raiz = Path(raiz)
        self.piezas = {}          # {"A": [Plantilla, ...], ...}
        self._cargar()

    def _cargar(self):
        for carpeta in sorted(self.raiz.iterdir()):
            if not carpeta.is_dir():
                continue
            # pieza_A -> "A"
            nombre = carpeta.name.split("_")[-1].upper()
            vistas = []
            for img_path in sorted(carpeta.glob("*.jpg")):
                vistas.append(Plantilla(nombre, img_path))
            if vistas:
                self.piezas[nombre] = vistas
                print(f"[Registro] Pieza {nombre}: {len(vistas)} vistas cargadas")

    def categorias(self):
        return list(self.piezas.keys())