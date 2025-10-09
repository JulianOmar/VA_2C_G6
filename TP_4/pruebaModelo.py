from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO('runs/classify/train/weights/best.pt') # Ruta al modelo entrenado

# Realizar predicciones en nuevas imágenes
results = model.predict(source='nuevas_imagenes/',  # Carpeta con nuevas imágenes
                        imgsz=224,  # Tamaño de las imágenes
                        conf=0.25,  # Umbral de confianza
                        batch=16)  # Tamaño del lote
# Mostrar resultados
for result in results:
    print(result)  # Imprimir resultados de la predicción

