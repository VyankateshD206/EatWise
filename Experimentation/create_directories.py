import os

def create_dataset_structure():
    """Create the necessary directory structure for YOLO training"""
    # Use the dataset path as the base directory
    base_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    
    print(f"Creating directories in: {base_dir}")
    
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        print(f"Created base directory: {base_dir}")
    
    # Create directory structure for each split
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base_dir, split)
        
        # Check if directory exists
        if os.path.exists(split_dir):
            print(f"Directory already exists: {split_dir}")
        else:
            # Create split directory
            os.makedirs(split_dir, exist_ok=True)
            print(f"Created directory: {split_dir}")
        
        # Create images and labels subdirectories
        images_dir = os.path.join(split_dir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
            print(f"Created directory: {images_dir}")
        
        labels_dir = os.path.join(split_dir, 'labels')
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir, exist_ok=True)
            print(f"Created directory: {labels_dir}")
    
    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("1. Place your images in the respective dataset/train/valid/test/images folders")
    print("2. Run prepare_dataset.py to analyze your dataset and create a YAML configuration")
    print("3. If needed, use excel_to_coco.py to convert your Excel annotations to COCO format")
    print("4. Use coco_to_yolo.py to convert COCO annotations to YOLO format")
    print("5. Finally, run train_yolo.py to train your model")

if __name__ == "__main__":
    create_dataset_structure()
