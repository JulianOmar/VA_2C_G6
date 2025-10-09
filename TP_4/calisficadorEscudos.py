from ultralytics import YOLO

# Usar YOLO para CLASIFICACIÓN (no detección)
model = YOLO('yolov8n-cls.pt')  # Modelo pre-entrenado para clasificación

# Entrenar para clasificación de escudos
results = model.train(
    data='Dataset',  # Estructura de carpetas por clase
    epochs=100,
    imgsz=224,
    patience=10,
    batch=32
)