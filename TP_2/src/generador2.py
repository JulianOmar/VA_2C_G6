"""Este módulo proporciona funciones para la generación de un dataset de invariantes de Hu a partir de imágenes de formas geométricas. 
Incluye utilidades para aumentar datos mediante transformaciones como rotación, escalado, adición de ruido y ajuste de brillo/contraste, 
así como el cálculo de invariantes de Hu para cada imagen original y aumentada. 
El resultado se guarda en un archivo CSV para su uso en tareas de clasificación.
Funciones principales:
- rotar_image: Rota una imagen en torno a su centro.
- escalar_imagen: Escala una imagen según un factor dado.
- agregar_ruido: Añade ruido gaussiano a la imagen.
- ajustar_brillo_contraste: Modifica el brillo y contraste de la imagen.
- calcular_invariantes_hu: Calcula los invariantes de Hu de una imagen.
- aplicar_aumentos_combinados: Aplica aumentos aleatorios combinados a una imagen.
El script recorre carpetas de clases ('circulo', 'cuadrado', 'triangulo'), procesa cada imagen y sus variantes aumentadas, 
calcula sus invariantes de Hu y guarda los resultados en un archivo CSV.
"""
import random
import cv2 as cv
import numpy as np
import pandas as pd # type: ignore
from pathlib import Path

def rotar_image(img, angulo):
    """Rota una imagen en torno a su centro"""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angulo, 1.0)
    return cv.warpAffine(img, M, (w, h))

def escalar_imagen(img, scale_factor):
    """Escala la imagen (aumenta o reduce el tamaño)"""
    return cv.resize(img, None, fx=scale_factor, fy=scale_factor)

def agregar_ruido(img):
    """Agrega ruido gaussiano."""
    ruido = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv.add(img, ruido)

def ajustar_brillo_contraste(img, brillo=1.0, contraste=0):
    """Modifica brillo (beta) y contraste (alpha)"""
    return cv.convertScaleAbs(img, alpha=brillo, beta=contraste)

def calcular_invariantes_hu(img_path=None, img=None):
    """Procesa una imagen y retorna sus invariantes de Hu:
    - img_path: Ruta de la imagen
    - img: Imagen en formato numpy
    (solo recibe uno de los parámetros)
    """

    if img_path is not None:
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if img is None:
            return None  # Error al leer la imagen
    
    if img is None:
        return None  # Si no se recibe img ni img_path, es un error

    _, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None  # No hay contornos

    c = max(contours, key=cv.contourArea)
    moments = cv.moments(c)
    hu = cv.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)  # Escala logarítmica
    return hu_log

def aplicar_aumentos_combinados(img, cantidad=5):    
    versiones = []

    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        return None

    for _ in range(cantidad):
        copia = img.copy()
        if random.random() < 0.5:
            angulo = random.choice([-30, -20, -10, 10, 20, 30])
            copia = rotar_image(copia, angulo)
        if random.random() < 0.3:
            factor = random.choice([0.6, 0.9, 1.1])
            copia = escalar_imagen(copia, factor)
        if random.random() < 0.4:
            copia = agregar_ruido(copia)
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-30, 30)
            copia = ajustar_brillo_contraste(copia, alpha, beta)
        versiones.append(copia)
    return versiones

etiquetas = {
    'circulo': 1,
    'cuadrado': 2,
    'triangulo': 3
}

data = []

# Itero por cada carpeta (clase)
base_path = Path('./formas')

for clase in etiquetas:
    carpeta = base_path / clase
    if not carpeta.exists():
        continue
    
    for img_path in carpeta.glob('*.png'):
        label = etiquetas[clase]
        
        hu = calcular_invariantes_hu(img_path=img_path)
        if hu is not None:
            data.append(list(hu) + [label])
        
        aumentadas = aplicar_aumentos_combinados(img_path, cantidad=5)
        for variante in aumentadas:
            hu = calcular_invariantes_hu(img=variante)
            if hu is not None:
                data.append(list(hu) + [label])

# Guardo CSV
df = pd.DataFrame(data, columns=[f'hu{i+1}' for i in range(7)] + ['etiqueta'])
df.to_csv('hu_dataset.csv', index=False)
print("Dataset generado con", len(df), "muestras.")
