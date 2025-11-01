import albumentations as A
import cv2
import os
from pathlib import Path
import numpy as np

def check_dataset_structure(input_dir):
    """Verifica la estructura del dataset y muestra información"""
    print(f"\n=== Verificando estructura de: {input_dir} ===")
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ La carpeta {input_dir} NO existe")
        return False
    
    print(f"✓ Carpeta principal existe")
    
    # Buscar imágenes en diferentes ubicaciones
    possible_locations = [
        input_path / 'images',
        input_path / 'train' / 'images',
        input_path,
    ]
    
    for loc in possible_locations:
        if loc.exists():
            images = list(loc.glob('*.jpg')) + list(loc.glob('*.png')) + list(loc.glob('*.jpeg'))
            if images:
                print(f"\n✓ Encontradas {len(images)} imágenes en: {loc}")
                print(f"  Ejemplos: {[img.name for img in images[:3]]}")
                
                # Verificar labels
                possible_label_dirs = [
                    loc.parent / 'labels',
                    loc.parent / 'labels' / 'train',
                    loc / 'labels',
                ]
                
                for label_dir in possible_label_dirs:
                    if label_dir.exists():
                        labels = list(label_dir.glob('*.txt'))
                        print(f"✓ Encontradas {len(labels)} etiquetas en: {label_dir}")
                        return True
    
    print("\n❌ No se encontraron imágenes")
    print("\nEstructura actual:")
    for item in input_path.rglob('*'):
        if item.is_file():
            print(f"  {item.relative_to(input_path)}")
    
    return False

def augment_dataset(input_dir, output_dir, augmentations_per_image=50):
    """
    Aumenta el dataset de imágenes con sus anotaciones YOLO
    
    Args:
        input_dir: Carpeta con images/ y labels/
        output_dir: Carpeta de salida
        augmentations_per_image: Número de variaciones por imagen
    """
    
    # Definir transformaciones agresivas para carne (CORREGIDAS)
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.3),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-45, 45),
            p=0.5
        ),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.3),
            A.Sharpen(p=0.3),
            A.Emboss(p=0.3),
        ], p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(p=0.2),
        A.RandomShadow(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.3
        ),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Detectar estructura automáticamente
    input_path = Path(input_dir)
    
    # Buscar carpeta de imágenes (estructura: images/train/ o train/images/)
    possible_image_dirs = [
        input_path / 'images' / 'train',  # Nueva: images/train/
        input_path / 'images' / 'val',    # Nueva: images/val/
        input_path / 'images',
        input_path / 'train' / 'images',
        input_path,
    ]
    
    input_images = None
    for img_dir in possible_image_dirs:
        if img_dir.exists():
            images_found = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
            if images_found:
                input_images = img_dir
                break
    
    if input_images is None:
        print("❌ No se encontró carpeta de imágenes válida")
        return
    
    # Buscar carpeta de labels (estructura: labels/train/ o train/labels/)
    # Inferir el split (train/val) desde la ruta de imágenes
    if 'train' in str(input_images):
        split = 'train'
    elif 'val' in str(input_images):
        split = 'val'
    else:
        split = None
    
    possible_label_dirs = [
        input_path / 'labels' / split if split else None,  # Nueva: labels/train/ o labels/val/
        input_images.parent.parent / 'labels' / split if split else None,
        input_images.parent / 'labels',
        input_images / 'labels',
    ]
    
    input_labels = None
    for lbl_dir in [d for d in possible_label_dirs if d]:
        if lbl_dir.exists():
            input_labels = lbl_dir
            break
    
    if input_labels is None:
        print("❌ No se encontró carpeta de labels")
        return
    
    print(f"\n✓ Usando:")
    print(f"  Imágenes: {input_images}")
    print(f"  Labels: {input_labels}")
    
    # Mantener la estructura: output_dir/images/train y output_dir/labels/train
    # Detectar si es train o val
    if 'train' in str(input_images):
        split = 'train'
    elif 'val' in str(input_images):
        split = 'val'
    else:
        split = 'train'  # default
    
    output_images = Path(output_dir) / 'images' / split
    output_labels = Path(output_dir) / 'labels' / split
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"  Salida imágenes: {output_images}")
    print(f"  Salida labels: {output_labels}")
    
    image_files = list(input_images.glob('*.jpg')) + list(input_images.glob('*.png')) + list(input_images.glob('*.jpeg'))
    
    if len(image_files) == 0:
        print("❌ No se encontraron archivos de imagen")
        return
    
    print(f"\n✓ Encontradas {len(image_files)} imágenes originales")
    print(f"  Generando {augmentations_per_image} augmentaciones por imagen...")
    
    total_generated = 0
    
    for img_path in image_files:
        print(f"\nProcesando: {img_path.name}")
        
        # Leer imagen
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ❌ No se pudo leer la imagen")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leer anotaciones YOLO
        label_path = input_labels / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"  ⚠️  No se encontró label: {label_path.name}")
            continue
        
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        if len(annotations) == 0:
            print(f"  ⚠️  Archivo de label vacío")
            continue
        
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
        
        # Guardar imagen original
        original_name = img_path.stem
        cv2.imwrite(str(output_images / f"{original_name}_orig.jpg"), 
                   cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        with open(output_labels / f"{original_name}_orig.txt", 'w') as f:
            for cls, bbox in zip(class_labels, bboxes):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")
        
        total_generated += 1
        
        # Generar augmentaciones
        successful_augs = 0
        for i in range(augmentations_per_image):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                
                if len(aug_bboxes) == 0:
                    continue
                
                # Guardar imagen aumentada
                aug_name = f"{original_name}_aug_{i:03d}.jpg"
                cv2.imwrite(str(output_images / aug_name), 
                           cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # Guardar anotaciones aumentadas
                with open(output_labels / f"{original_name}_aug_{i:03d}.txt", 'w') as f:
                    for cls, bbox in zip(aug_labels, aug_bboxes):
                        f.write(f"{cls} {' '.join(map(str, bbox))}\n")
                
                successful_augs += 1
                total_generated += 1
                    
            except Exception as e:
                print(f"  ⚠️  Error en augmentación {i}: {e}")
                continue
        
        print(f"  ✓ Generadas {successful_augs} augmentaciones exitosas")
    
    final_count = len(list(output_images.glob('*.jpg')))
    print(f"\n{'='*50}")
    print(f"✓ Dataset aumentado exitosamente!")
    print(f"  Imágenes originales: {len(image_files)}")
    print(f"  Total generadas: {final_count}")
    if len(image_files) > 0:
        print(f"  Factor de aumento: {final_count / len(image_files):.1f}x")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Configurar para tu estructura: images/train/ y labels/train/
    INPUT_DIR = "carneDataset"  # Carpeta raíz con images/ y labels/
    OUTPUT_DIR = "carneDataSet_Augmented_Large"  # Saldrá igual: images/ y labels/
    AUGMENTATIONS_PER_IMAGE = 100
    
    print("="*50)
    print("AUGMENTACIÓN DE DATASET DE CORTES DE CARNE")
    print("="*50)
    
    # Procesar train
    print("\n>>> Procesando TRAIN <<<")
    augment_dataset(INPUT_DIR, OUTPUT_DIR, AUGMENTATIONS_PER_IMAGE)
    
    # El script detectará automáticamente train/val y mantendrá la estructura
    print("\n✓ Augmentación completada!")
    print(f"\nEstructura de salida:")
    print(f"{OUTPUT_DIR}/")
    print(f"├── images/")
    print(f"│   ├── train/")
    print(f"│   └── val/")
    print(f"└── labels/")
    print(f"    ├── train/")
    print(f"    └── val/")