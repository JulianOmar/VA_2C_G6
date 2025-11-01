# train_detection.py
from ultralytics import YOLO
import os

def train_detection_model():
    """Entrena modelo YOLOv8 para detección de escudos"""
    
    # Cargar modelo pre-entrenado en COCO
    model = YOLO('yolov8m.pt')  # o 'yolov8s.pt', 'yolov8m.pt' para más precisión
    
    # Entrenar el modelo
    results = model.train(
        data='carneDataSet_Augmented_Large/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        lr0=0.01,
        lrf=0.01,
        optimizer='AdamW',
        weight_decay=0.0005,
        save=True,
        device='0',  # Usar GPU
        project='cortes_detection',
        name='yolov8m_train_augmented_large',
        exist_ok=True
    )
    
    return model, results

if __name__ == '__main__':
    # Opcional: para compatibilidad en Windows congelado (PyInstaller)
    from multiprocessing import freeze_support
    freeze_support()
    model, results = train_detection_model()


