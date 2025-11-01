# predict_detection.py
from ultralytics import YOLO
import cv2
import numpy as np

class CarneDetector:
    def __init__(self, model_path, confidence_thresh=0.60):
        self.model = YOLO(model_path)
        self.confidence_thresh = confidence_thresh
        self.class_names = self.model.names
    
    def detect_image(self, image_path, save_result=True):
        """Detección en una imagen"""
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_thresh,
            save=save_result,
            project='predictions',
            name='detection_results'
        )
        
        return self.process_detections(results[0])
    
    def detect_video(self, video_path, output_path=None):
        """Detección en video"""
        results = self.model.predict(
            source=video_path,
            conf=self.confidence_thresh,
            save=True,
            project='predictions',
            name='video_detection'
        )
        return results
    
    def detect_webcam(self):
        """Detección en tiempo real desde webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(
                source=frame,
                conf=self.confidence_thresh,
                verbose=False
            )
            
            # Mostrar resultados
            annotated_frame = results[0].plot()
            cv2.imshow('Corte detección', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_detections(self, result):
        """Procesa y muestra los resultados de detección"""
        detections = []
        
        print(f"\n=== DETECCIONES ENCONTRADAS ===")
        print(f"Imagen: {result.path}")
        print(f"Corte detectados: {len(result.boxes)}\n")
        
        for i, box in enumerate(result.boxes):
            detection = {
                'id': i,
                'class_id': int(box.cls[0]),
                'class_name': self.class_names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                'bbox_normalized': box.xyxyn[0].tolist()
            }
            detections.append(detection)
            
            print(f"Nombre: {i+1}:")
            print(f"  Nombre: {detection['class_name']}")
            print(f"  Confianza: {detection['confidence']:.2%}")
            print(f"  Posición: {detection['bbox']}")
            print()
        
        return {
            'image_path': result.path,
            'detections': detections,
            'total_detections': len(detections)
        }

# Ejemplo de uso
detector = CarneDetector('cortes_detection/yolov8m_corregido/weights/best.pt')

# Detección en imagen
#results = detector.detect_image('test_image.jpg')

# Detección en tiempo real (descomentar para usar)
detector.detect_webcam()