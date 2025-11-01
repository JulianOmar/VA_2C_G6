import os
def analyze_dataset_structure(dataset_path):
    """Analiza cómo se asignan las etiquetas automáticamente"""
    
    # Obtener lista de clases (nombres de carpetas)
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    # Crear mapeo automático
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    print("=== ESTRUCTURA DEL DATASET ===")
    print(f"Total de clases: {len(classes)}")
    print("\nMapeo de etiquetas:")
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        num_images = len([f for f in os.listdir(class_path) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"  {idx}: {class_name} ({num_images} imágenes)")
    
    return class_to_idx, idx_to_class

# Ejecutar análisis
dataset_path = "Dataset"
class_mapping, reverse_mapping = analyze_dataset_structure(dataset_path)