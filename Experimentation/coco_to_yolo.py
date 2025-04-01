import json
import os
from pathlib import Path
import shutil

def convert_coco_to_yolo(coco_file, output_dir, class_mapping=None):
    """
    Convert COCO annotations to YOLO format
    
    Args:
        coco_file: Path to COCO annotation file
        output_dir: Directory to save YOLO format annotations
        class_mapping: Optional dict mapping class names to indices
    """
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class mapping if not provided
    if class_mapping is None:
        categories = coco_data['categories']
        class_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    # Map image IDs to filenames
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Process each image
    for img_id, annotations in annotations_by_image.items():
        img_info = image_map[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Get base filename without extension
        filename = Path(img_info['file_name']).stem
        
        # Create YOLO format annotations
        yolo_lines = []
        for ann in annotations:
            # Get class index
            cat_id = ann['category_id']
            class_idx = class_mapping.get(cat_id, cat_id)
            
            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            
            # Convert to YOLO format: [class_idx, x_center, y_center, width, height]
            # Where x, y, width, height are normalized by image dimensions
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            
            # Add to lines
            yolo_lines.append(f"{class_idx} {x_center} {y_center} {width} {height}")
        
        # Write to file
        output_file = os.path.join(output_dir, f"{filename}.txt")
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    print(f"Converted {len(annotations_by_image)} images to YOLO format")

def main():
    # Use the dataset path as the base directory
    base_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    splits = ['train', 'valid', 'test']
    
    print(f"Looking for COCO annotations in: {base_dir}")
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} - directory not found: {split_dir}")
            continue
        
        coco_file = os.path.join(split_dir, 'annotations.json')
        if not os.path.exists(coco_file):
            print(f"Skipping {split} - no COCO annotations found: {coco_file}")
            continue
        
        # Create output directory
        yolo_dir = os.path.join(split_dir, 'labels')
        os.makedirs(yolo_dir, exist_ok=True)
        
        # Convert
        print(f"Converting {split} annotations...")
        convert_coco_to_yolo(coco_file, yolo_dir)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
