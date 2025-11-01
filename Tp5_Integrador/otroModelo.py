# train_detection_corregido.py
from ultralytics import YOLO
import os

def train_detection_model():
    """Entrena modelo YOLOv8 para detección de cortes de carne"""
    
    # Cargar modelo pre-entrenado
    model = YOLO('yolov8m.pt')
    
    # Entrenar el modelo CON HIPERPARÁMETROS CORREGIDOS
    results = model.train(
        data='carneDataSet_Augmented_Large/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        # ✅ HIPERPARÁMETROS CORREGIDOS:
        lr0=0.001,           # Learning rate inicial MÁS BAJO
        lrf=0.01,            # Learning rate final (1% del inicial)
        momentum=0.937,      # Momentum para SGD
        weight_decay=0.0005,
        warmup_epochs=3.0,   # Calentamiento gradual
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,            # Peso de la pérdida de bbox
        cls=0.5,            # Peso de la pérdida de clases
        dfl=1.5,            # Peso de la pérdida DFL
        optimizer='SGD',     # ✅ SGD funciona mejor para YOLO
        save=True,
        device='0',
        project='cortes_detection',
        name='yolov8m_corregido',
        exist_ok=True
    )
    
    return model, results

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    model, results = train_detection_model()