import os
import pandas as pd
import json
import shutil
from pathlib import Path

def load_classes_from_excel(excel_path):
    """Extract class names from your Excel annotation file"""
    df = pd.read_excel(excel_path)
    
    # This assumes your Excel file has a 'category_name' column
    # Adjust the column name as needed
    if 'category_name' in df.columns:
        classes = df['category_name'].unique().tolist()
    elif 'class' in df.columns:
        classes = df['class'].unique().tolist()
    else:
        # Try to find a column that might contain class names
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['class', 'category', 'label']):
                classes = df[col].unique().tolist()
                break
        else:
            raise ValueError("Could not find class names in Excel file. Please specify the column name.")
    
    return classes

def verify_dataset_structure(base_dir):
    """Verify and print stats about your dataset structure"""
    subdirs = ['train', 'valid', 'test']
    stats = {}
    
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)  # Updated path
        images_path = os.path.join(dir_path, 'images')
        labels_path = os.path.join(dir_path, 'labels')
        
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_path} does not exist!")
            continue
            
        # Check for COCO annotations
        coco_path = os.path.join(dir_path, 'annotations.json')
        has_coco = os.path.exists(coco_path)
        
        # Count images
        img_count = 0
        if os.path.exists(images_path):
            img_count = len([f for f in os.listdir(images_path) 
                           if os.path.isfile(os.path.join(images_path, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        stats[subdir] = {
            'image_count': img_count,
            'has_coco_annotations': has_coco
        }
    
    return stats

def create_dataset_structure(base_dir):
    """Create the dataset directory structure if it doesn't exist"""
    # Create subdirectories for each split
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base_dir, split)  # Updated path
        os.makedirs(split_dir, exist_ok=True)
        
        # Create images and labels directories
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
    
    print("Created dataset directory structure.")
    return base_dir

def create_dataset_yaml(base_dir, classes):
    """Create a YAML file for YOLOv8 training configuration"""
    yaml_content = {
        'path': base_dir,  # Updated path
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(classes),  # number of classes
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    yaml_path = os.path.join(base_dir, 'dataset.yaml')
    
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset config at {yaml_path}")
    return yaml_path

def extract_annotations_from_excel(excel_path, dataset_dir):
    """Extract annotations from Excel and organize into the dataset structure"""
    df = pd.read_excel(excel_path)
    
    # This is a placeholder function - you'll need to customize based on your Excel structure
    print("\nTo complete the dataset setup:")
    print("1. Place your images in the respective dataset/[train|valid|test]/images directories")
    print("2. If you have COCO format annotations, place annotations.json in each dataset/[train|valid|test] directory")
    print("3. Or use the coco_to_yolo.py script to convert annotations to YOLO format\n")
    
    return True

def main():
    # Use the dataset path as the base directory
    base_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    parent_dir = os.path.dirname(base_dir)
    
    # Create the base_dir if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        print(f"Created directory: {base_dir}")
    
    # Ask user for the Excel file path
    excel_path = input("Enter path to your Excel annotations file: ")
    
    try:
        # Extract classes from Excel
        classes = load_classes_from_excel(excel_path)
        print(f"Found {len(classes)} classes: {', '.join(classes)}")
        
        # Check if dataset structure exists, if not create it
        stats = verify_dataset_structure(base_dir)
        
        # Print the directory structure status
        print("\nChecking directories:")
        for split in ['train', 'valid', 'test']:
            dir_path = os.path.join(base_dir, split)
            print(f"  {split}: {'Exists' if os.path.exists(dir_path) else 'Missing'}")
            
            # Check subdirectories if they exist
            if os.path.exists(dir_path):
                images_dir = os.path.join(dir_path, 'images')
                labels_dir = os.path.join(dir_path, 'labels')
                print(f"    images/: {'Exists' if os.path.exists(images_dir) else 'Missing'}")
                print(f"    labels/: {'Exists' if os.path.exists(labels_dir) else 'Missing'}")
        
        # Check if any of the directories are missing
        missing_dirs = any(not os.path.exists(os.path.join(base_dir, split)) 
                          for split in ['train', 'valid', 'test'])
        
        if missing_dirs:
            create_choice = input("\nDataset structure is incomplete. Would you like to create it now? (y/n): ")
            if create_choice.lower() == 'y':
                dataset_dir = create_dataset_structure(base_dir)
                # Offer to help organize data
                organize_choice = input("Would you like guidance on organizing your data? (y/n): ")
                if organize_choice.lower() == 'y':
                    extract_annotations_from_excel(excel_path, dataset_dir)
        
        # Create YAML config file - save both in dataset folder and parent folder
        yaml_path = create_dataset_yaml(base_dir, classes)
        
        # Also create a copy in the parent directory for compatibility
        yaml_content = {
            'path': base_dir,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': {i: name for i, name in enumerate(classes)}
        }
        
        parent_yaml_path = os.path.join(parent_dir, 'dataset.yaml')
        import yaml
        with open(parent_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Also created dataset config at {parent_yaml_path}")
        
        print("\nDataset preparation complete!")
        print("Next steps:")
        print("1. Make sure your images are in dataset/[train|valid|test]/images directories")
        print("2. Ensure your annotations are properly organized")
        print("3. Run: python train_yolo.py")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

if __name__ == "__main__":
    main()
