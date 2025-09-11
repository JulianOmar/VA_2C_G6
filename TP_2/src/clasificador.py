import cv2 as cv
import numpy as np
import joblib 
"""Clasificador de formas en tiempo real usando la cámara web y un modelo pre-entrenado.

Este script captura video desde la cámara web predeterminada, procesa cada cuadro para detectar formas geométricas
(círculos, cuadrados, triángulos) y las clasifica utilizando un modelo de machine learning entrenado con momentos de Hu.
El usuario puede ajustar parámetros de umbralización y operaciones morfológicas mediante barras de control de OpenCV.

Características principales:
- Carga un clasificador pre-entrenado desde 'modelo_entrenado.joblib'.
- Utiliza los momentos de Hu como características para la clasificación de formas.
- Permite el ajuste en tiempo real del umbral de binarización, tamaño del kernel morfológico y (opcionalmente) el umbral de matchShapes.
- Detecta contornos en la imagen binarizada, filtra por área y clasifica cada forma detectada.
- Dibuja rectángulos y etiquetas de la forma predicha sobre el cuadro original.
- Muestra una ventana combinada con el cuadro anotado y la máscara binaria.

Barras de control:
- "Umbral_binarizacion": Valor de umbral para la binarización.
- "Tam_estructura": Tamaño del kernel para la apertura morfológica.
- "Umbral_matchShapes": (No utilizado en el código actual) Previsto para el umbral de comparación de formas.

Presione 'Esc' para salir de la aplicación.

Dependencias:

Nota:
- Asegúrese de que 'modelo_entrenado.joblib' exista en el directorio de trabajo.
- El clasificador debe estar entrenado para reconocer los momentos de Hu de las formas deseadas.
"""

# Cargo el modelo entrenado
modelo = joblib.load('modelo_entrenado.joblib')

etiquetas = {
    1: "circulo",
    2: "cuadrado",
    3: "triangulo"
}

# Barra de opciones
cv.namedWindow("Ajustes")
cv.createTrackbar("Umbral_binarizacion", "Ajustes", 100, 255, lambda x: None)
cv.createTrackbar("Tam_estructura", "Ajustes", 1, 20, lambda x: None)
cv.createTrackbar("Umbral_matchShapes", "Ajustes", 20, 100, lambda x: None)

webcam = cv.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Obtenemos el umbral desde la barra
    thresh_val = cv.getTrackbarPos("Umbral_binarizacion", "Ajustes")
    _, binary = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY_INV)

    # Acá lo mismo pero con el tamaño de la estructura para operaciones morfológicas
    tam_estruc = cv.getTrackbarPos("Tam_estructura", "Ajustes")
    if tam_estruc < 1:
        tam_estruc = 1
    kernel = np.ones((tam_estruc, tam_estruc), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    # Detectar contornos
    contornos, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        area = cv.contourArea(contorno)
        if area < 500:  # Ignorar contornos muy pequeños
            continue

        # Calcular momentos e invariantes de Hu
        momentos = cv.moments(contorno)
        hu = cv.HuMoments(momentos)
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        vector = hu.flatten().reshape(1, -1)

        # Predecir figura
        prediccion = modelo.predict(vector)[0]
        nombre_figura = etiquetas.get(prediccion, "Desconocido")

        # Dibujar contorno y texto
        x, y, w, h = cv.boundingRect(contorno)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, nombre_figura, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Acá juntamos las 2 ventanas y cambiamos el tamaño para ajustarlo mejor a la pantalla 
    bin_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)  # Convertir a color para concatenar
    combinada = cv.hconcat([frame, bin_color])
    escala = 0.5
    combinada = cv.resize(combinada, (int(combinada.shape[1] * escala), int(combinada.shape[0] * escala)))

    cv.imshow("Vista combinada", combinada)

    if cv.waitKey(1) & 0xFF == 27: # Tecla 'Esc' para salir
        break

webcam.release()
cv.destroyAllWindows()
