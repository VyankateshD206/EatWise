import os
import json
import pandas as pd
from pathlib import Path
import datetime

def excel_to_coco(excel_path, image_dir, output_file, split_column=None, split_value=None):
    """
    Convert Excel annotations to COCO format
    
    Args:
        excel_path: Path to Excel file with annotations
        image_dir: Directory containing the images
        output_file: Path to save the COCO JSON file
        split_column: Column name to filter by (e.g., 'split')
        split_value: Value to filter for (e.g., 'train', 'valid', 'test')
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Filter by split if specified
    if split_column and split_value:
        if split_column in df.columns:
            df = df[df[split_column] == split_value]
        else:
            print(f"Warning: Column '{split_column}' not found in Excel file. Using all data.")
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Converted from Excel annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get unique categories/classes
    class_column = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['class', 'category', 'label']):
            class_column = col
            break
    
    if not class_column:
        raise ValueError("Could not find class/category column in Excel file")
    
    # Create category entries
    categories = df[class_column].unique()
    for i, cat_name in enumerate(categories):
        coco_data["categories"].append({
            "id": i + 1,
            "name": cat_name,
            "supercategory": "none"
        })
    
    # Map category names to IDs
    cat_name_to_id = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
    
    # Detect required columns
    required_columns = ["image_file", class_column]
    bbox_columns = []
    
    # Check for various possible bbox column naming patterns
    bbox_patterns = [
        ["bbox_x", "bbox_y", "bbox_width", "bbox_height"],
        ["x", "y", "width", "height"],
        ["x1", "y1", "x2", "y2"],
        ["xmin", "ymin", "xmax", "ymax"]
    ]
    
    for pattern in bbox_patterns:
        if all(col in df.columns for col in pattern):
            bbox_columns = pattern
            break
    
    if not bbox_columns:
        print("Could not detect bounding box columns. Please enter the column names:")
        x_col = input("X coordinate column name: ")
        y_col = input("Y coordinate column name: ")
        w_col = input("Width column name (or leave empty if using x2/y2): ")
        h_col = input("Height column name (or leave empty if using x2/y2): ")
        
        if w_col and h_col:
            bbox_columns = [x_col, y_col, w_col, h_col]
        else:
            x2_col = input("X2 coordinate column name: ")
            y2_col = input("Y2 coordinate column name: ")
            bbox_columns = [x_col, y_col, x2_col, y2_col]
    
    # Flag for conversion from x1,y1,x2,y2 to x,y,w,h
    convert_coords = len(bbox_columns) == 4 and bbox_columns[2] in ["x2", "xmax"] and bbox_columns[3] in ["y2", "ymax"]
    
    # Process images and annotations
    image_id = 1
    annotation_id = 1
    processed_images = set()
    
    for idx, row in df.iterrows():
        image_file = row.get("image_file") or row.get("filename") or row.get("file") or row.get("image")
        if not image_file:
            print(f"Warning: No image filename in row {idx+1}, skipping")
            continue
        
        # Add image if not already added
        if image_file not in processed_images:
            # Try to get image dimensions
            image_path = os.path.join(image_dir, image_file)
            width = row.get("width") or row.get("image_width") or 0
            height = row.get("height") or row.get("image_height") or 0
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "license": 1,
                "file_name": image_file,
                "height": height,
                "width": width,
                "date_captured": ""
            })
            processed_images.add(image_file)
        
        # Get bounding box coordinates
        try:
            if convert_coords:
                # Convert from x1,y1,x2,y2 to x,y,w,h
                x1 = float(row[bbox_columns[0]])
                y1 = float(row[bbox_columns[1]])
                x2 = float(row[bbox_columns[2]])
                y2 = float(row[bbox_columns[3]])
                
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
            else:
                # Already in x,y,w,h format
                x = float(row[bbox_columns[0]])
                y = float(row[bbox_columns[1]])
                w = float(row[bbox_columns[2]])
                h = float(row[bbox_columns[3]])
        except (ValueError, KeyError) as e:
            print(f"Warning: Error parsing bbox in row {idx+1}: {str(e)}")
            continue
        
        # Get category
        category_name = row[class_column]
        category_id = cat_name_to_id.get(category_name)
        if not category_id:
            print(f"Warning: Unknown category '{category_name}' in row {idx+1}")
            continue
        
        # Add annotation
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation": [],
            "iscrowd": 0
        })
        
        annotation_id += 1
        
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Converted {len(processed_images)} images and {annotation_id-1} annotations to COCO format")
    print(f"Saved to {output_file}")
    
    return len(processed_images), annotation_id-1

def main():
    print("Excel to COCO Converter")
    excel_path = input("Enter path to Excel annotations file: ")
    
    # Ask which split to process
    split_choice = input("Which data split to process? (train/valid/test/all): ").lower()
    
    # Use the dataset path as the base directory
    base_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    
    if split_choice == "all":
        splits = ["train", "valid", "test"]
    elif split_choice in ["train", "valid", "test"]:
        splits = [split_choice]
    else:
        print("Invalid choice. Using 'all'")
        splits = ["train", "valid", "test"]
    
    split_column = input("Enter column name that specifies the data split (or leave empty if none): ")
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        split_dir = os.path.join(base_dir, split)
        image_dir = os.path.join(split_dir, "images")
        output_file = os.path.join(split_dir, "annotations.json")
        
        # Create directories if they don't exist
        if not os.path.exists(split_dir):
            print(f"Creating directory: {split_dir}")
            os.makedirs(split_dir, exist_ok=True)
            
        if not os.path.exists(image_dir):
            print(f"Creating directory: {image_dir}")
            os.makedirs(image_dir, exist_ok=True)
            print(f"Warning: {image_dir} is empty. Please add images before processing.")
            continue
        
        # Process the Excel file for this split
        try:
            if split_column:
                split_value = split
                images, annotations = excel_to_coco(excel_path, image_dir, output_file, split_column, split_value)
            else:
                # If no split column, use all data for each split (user should select only one split)
                images, annotations = excel_to_coco(excel_path, image_dir, output_file)
                
            print(f"Processed {split} split: {images} images, {annotations} annotations")
            
        except Exception as e:
            print(f"Error processing {split} split: {str(e)}")
    
    print("\nConversion complete!")
    print("Next steps:")
    print("1. Review the generated annotations.json files")
    print("2. Run prepare_dataset.py to verify the dataset")
    print("3. Train your model with train_yolo.py")

if __name__ == "__main__":
    main()
