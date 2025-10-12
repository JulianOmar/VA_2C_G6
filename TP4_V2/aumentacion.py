# massive_augmentation.py
import albumentations as A
import cv2
import os
from pathlib import Path

class MassiveAugmentor:
    def __init__(self, original_dataset_path, output_path):
        self.original_path = Path(original_dataset_path)
        self.output_path = Path(output_path)
        self.augmentations = self.get_augmentation_pipeline()
    
    def get_augmentation_pipeline(self):
        """Pipeline de aumentación masiva"""
        return A.Compose([
            # Transformaciones geométricas
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=30, 
                p=0.8
            ),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(0.1, 0.1),
                rotate=(-30, 30),
                shear=(-10, 10),
                p=0.7
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Transformaciones de color
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=0.4
            ),
            
            # Ruido y efectos
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.MedianBlur(blur_limit=5, p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.2
            ),
            
            # Transformaciones de calidad
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                angle_lower=0.5,
                p=0.1
            ),
            
            # Distorsiones
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.1
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.2
            ),
            
            # Modificaciones de canal
            A.ChannelShuffle(p=0.1),
            A.InvertImg(p=0.1),
            A.ToGray(p=0.1),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.3
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment_image(self, image, bboxes, class_labels, num_augmentations=20):
        """Genera múltiples versiones aumentadas de una imagen"""
        augmented_data = []
        
        for i in range(num_augmentations):
            try:
                augmented = self.augmentations(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                augmented_data.append({
                    'image': augmented['image'],
                    'bboxes': augmented['bboxes'],
                    'class_labels': augmented['class_labels']
                })
            except Exception as e:
                print(f"Error en aumentación {i}: {e}")
                continue
        
        return augmented_data
    
    def process_dataset(self, augmentations_per_image=50):
        """Procesa todo el dataset aplicando aumentación masiva"""
        # Crear estructura de directorios
        (self.output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        
        # Procesar cada imagen original
        original_images_dir = self.original_path / 'images' / 'train'
        original_labels_dir = self.original_path / 'labels' / 'train'
        
        image_files = [f for f in os.listdir(original_images_dir) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        total_original = len(image_files)
        total_augmented = 0
        
        print(f"🔧 Iniciando aumentación masiva...")
        print(f"   Imágenes originales: {total_original}")
        print(f"   Aumentaciones por imagen: {augmentations_per_image}")
        print(f"   Total esperado: {total_original * augmentations_per_image}")
        
        for img_file in image_files:
            # Cargar imagen original
            img_path = original_images_dir / img_file
            label_path = original_labels_dir / img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                continue
            
            # Cargar anotaciones originales
            bboxes = []
            class_labels = []
            img_height, img_width = image.shape[:2]
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            
            # Generar versiones aumentadas
            augmented_versions = self.augment_image(
                image, bboxes, class_labels, augmentations_per_image
            )
            
            # Guardar versiones aumentadas
            base_name = img_file.split('.')[0]
            
            for i, aug_data in enumerate(augmented_versions):
                # Guardar imagen aumentada
                aug_image_name = f"{base_name}_aug_{i:03d}.jpg"
                aug_image_path = self.output_path / 'images' / 'train' / aug_image_name
                
                aug_image_bgr = cv2.cvtColor(aug_data['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_image_path), aug_image_bgr)
                
                # Guardar anotaciones aumentadas
                aug_label_name = f"{base_name}_aug_{i:03d}.txt"
                aug_label_path = self.output_path / 'labels' / 'train' / aug_label_name
                
                with open(aug_label_path, 'w') as f:
                    for bbox, class_id in zip(aug_data['bboxes'], aug_data['class_labels']):
                        if all(0 <= coord <= 1 for coord in bbox):  # Validar coordenadas
                            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                total_augmented += 1
            
            print(f"✅ {img_file} → {len(augmented_versions)} aumentaciones")
        
        print(f"\n🎉 Aumentación completada!")
        print(f"   Total de imágenes generadas: {total_augmented}")
        
        # Copiar data.yaml
        original_yaml = self.original_path / 'data.yaml'
        if original_yaml.exists():
            import shutil
            shutil.copy2(original_yaml, self.output_path / 'data.yaml')

# USO: Aumentar dataset de 18 a 900+ imágenes
augmentor = MassiveAugmentor('detection_dataset', 'augmented_dataset')
augmentor.process_dataset(augmentations_per_image=50)  # 18 × 50 = 900 imágenes