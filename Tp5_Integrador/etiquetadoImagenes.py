# annotate_detection.py
import cv2
import os
import numpy as np

class DetectionAnnotator:
    def __init__(self, images_dir, labels_dir, class_names):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.current_image = None
        self.annotations = []
        self.current_class = 0
        self.drawing = False
        self.start_x, self.start_y = -1, -1

        # Añadido: escala para mostrar imagen y almacenar coordenadas correctas
        self.display_image = None
        self.scale = 1.0

    def mouse_callback(self, event, x, y, flags, param):
        """Maneja eventos del mouse para dibujar bounding boxes"""
        # x,y vienen en coordenadas de la ventana (imagen escalada). Convertir a coords originales.
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = orig_x, orig_y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.display_image is not None:
                # Dibujar en la copia escalada para mostrar feedback interactivo
                temp_disp = self.display_image.copy()
                disp_start = (int(self.start_x * self.scale), int(self.start_y * self.scale))
                disp_cur = (x, y)
                cv2.rectangle(temp_disp, disp_start, disp_cur, (0, 255, 0), 2)
                cv2.putText(temp_disp, self.class_names[self.current_class],
                           (disp_start[0], disp_start[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Annotator', temp_disp)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_x, end_y = orig_x, orig_y

            # Asegurar coordenadas válidas
            x1 = min(self.start_x, end_x)
            y1 = min(self.start_y, end_y)
            x2 = max(self.start_x, end_x)
            y2 = max(self.start_y, end_y)

            # Solo guardar si el bbox tiene tamaño suficiente
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                self.annotations.append((x1, y1, x2, y2, self.current_class))

            self.redraw_annotations()
    
    def redraw_annotations(self):
        """Redibuja todas las anotaciones en la imagen"""
        if self.display_image is None:
            return

        temp_disp = self.display_image.copy()
        for (x1, y1, x2, y2, class_id) in self.annotations:
            # Escalar bbox para mostrar en la imagen escalada
            dx1, dy1 = int(x1 * self.scale), int(y1 * self.scale)
            dx2, dy2 = int(x2 * self.scale), int(y2 * self.scale)
            cv2.rectangle(temp_disp, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            cv2.putText(temp_disp, f"{class_id}:{self.class_names[class_id]}",
                       (dx1, dy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Annotator', temp_disp)
    
    def convert_to_yolo_format(self, bbox, img_width, img_height):
        """Convierte bbox a formato YOLO (normalizado)"""
        x1, y1, x2, y2, class_id = bbox
        
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return class_id, x_center, y_center, width, height
    
    def save_annotations(self, image_filename, img_shape):
        """Guarda anotaciones en formato YOLO"""
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for bbox in self.annotations:
                class_id, x_center, y_center, width, height = self.convert_to_yolo_format(bbox, img_shape[1], img_shape[0])
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def annotate_images(self):
        """Herramienta interactiva de anotación"""
        image_files = [f for f in os.listdir(self.images_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        print(f"Encontradas {len(image_files)} imágenes para anotar")
        print("Controles:")
        print("0-9: Cambiar clase actual (primeras 10 clases)")
        print("a-z: Cambiar clase actual (clases 10-35)")
        print("+: Siguiente clase")
        print("-: Clase anterior")
        print("S: Guardar y siguiente imagen")
        print("D: Eliminar última anotación")
        print("L: Listar todas las clases")
        print("Q: Salir")

        for img_file in image_files:
            self.annotations = []
            image_path = os.path.join(self.images_dir, img_file)
            self.current_image = cv2.imread(image_path)

            if self.current_image is None:
                continue

            # Preparar ventana escalada y callback
            cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)

            # Calcular escala para mostrar (mantener aspecto, limitar tamaño máximo)
            max_dim = 800
            h, w = self.current_image.shape[:2]
            self.scale = min(max_dim / w, max_dim / h, 1.0)
            self.display_image = cv2.resize(self.current_image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

            cv2.setMouseCallback('Annotator', self.mouse_callback)

            print(f"\nAnotando: {img_file}")
            print(f"Clase actual: {self.current_class} - {self.class_names[self.current_class]}")

            while True:
                self.redraw_annotations()
                key = cv2.waitKey(20) & 0xFF

                # Cambiar clase con teclas 0-9 (primeras 10 clases)
                if ord('0') <= key <= ord('9'):
                    class_idx = key - ord('0')
                    if class_idx < len(self.class_names):
                        self.current_class = class_idx
                        print(f"Clase cambiada a: {self.current_class} - {self.class_names[self.current_class]}")

                # Cambiar clase con teclas a-z (clases 10-35)
                elif ord('a') <= key <= ord('z'):
                    class_idx = (key - ord('a')) + 10
                    if class_idx < len(self.class_names):
                        self.current_class = class_idx
                        print(f"Clase cambiada a: {self.current_class} - {self.class_names[self.current_class]}")

                # Siguiente clase
                elif key == ord('+') or key == ord('='):
                    self.current_class = (self.current_class + 1) % len(self.class_names)
                    print(f"Clase cambiada a: {self.current_class} - {self.class_names[self.current_class]}")

                # Clase anterior
                elif key == ord('-'):
                    self.current_class = (self.current_class - 1) % len(self.class_names)
                    print(f"Clase cambiada a: {self.current_class} - {self.class_names[self.current_class]}")

                # Listar todas las clases
                elif key == ord('L'):
                    print("\nLista de clases disponibles:")
                    for i, name in enumerate(self.class_names):
                        print(f"  {i}: {name}")

                # Guardar y siguiente
                elif key == ord('S'):
                    self.save_annotations(img_file, self.current_image.shape)
                    print(f"Guardadas {len(self.annotations)} anotaciones")
                    break
                elif key == ord('N'):
                    break;

                # Eliminar última anotación
                elif key == ord('D'):
                    if self.annotations:
                        self.annotations.pop()
                        print("Última anotación eliminada")

                # Salir
                elif key == ord('Q'):
                    cv2.destroyAllWindows()
                    return

            cv2.destroyAllWindows()

CLASS_NAMES = [
    'Asado', 'Vacio', 'Matambre',   
]

# Crear directorios si no existen
os.makedirs('carneDataset/images/val', exist_ok=True)
os.makedirs('carneDataset/labels/val', exist_ok=True)

annotator = DetectionAnnotator(
    images_dir='carneDataset/images/val',
    labels_dir='carneDataset/labels/val',
    class_names=CLASS_NAMES
)
annotator.annotate_images()