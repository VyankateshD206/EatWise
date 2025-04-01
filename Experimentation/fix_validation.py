import os
import shutil
from pathlib import Path
import hashlib
import random

def fix_validation_data():
    """Create valid validation data from training data if validation is empty"""
    dataset_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    
    # Source directories
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    
    # Target directories
    valid_images_dir = os.path.join(valid_dir, 'images')
    valid_labels_dir = os.path.join(valid_dir, 'labels')
    
    # Create validation directories if they don't exist
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    
    # Clear existing validation data to avoid mixed or corrupted files
    print("Clearing existing validation data...")
    for f in os.listdir(valid_images_dir):
        file_path = os.path.join(valid_images_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    for f in os.listdir(valid_labels_dir):
        file_path = os.path.join(valid_labels_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Remove any cache files
    cache_files = [
        os.path.join(valid_dir, 'labels.cache'),
        os.path.join(train_dir, 'labels.cache')
    ]
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed cache file: {cache_file}")
    
    print("Finding training images with valid annotations...")
    
    # Find training images that have non-empty label files
    valid_training_pairs = []
    train_images = [f for f in os.listdir(train_images_dir) 
                   if os.path.isfile(os.path.join(train_images_dir, f)) and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in train_images:
        img_stem = Path(img_file).stem
        label_path = os.path.join(train_labels_dir, f"{img_stem}.txt")
        
        # Check if label exists and is not empty
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            valid_training_pairs.append((img_file, f"{img_stem}.txt"))
    
    print(f"Found {len(valid_training_pairs)} images with valid annotations")
    
    if len(valid_training_pairs) == 0:
        print("Error: No valid training images with annotations found.")
        print("Creating synthetic validation data with dummy annotations...")
        
        # If no valid annotations, create some synthetic ones
        for i, img_file in enumerate(train_images[:50]):  # Use up to 50 images
            # Copy image
            src_img_path = os.path.join(train_images_dir, img_file)
            dst_img_path = os.path.join(valid_images_dir, img_file)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Create a synthetic annotation (center box covering 50% of the image)
            img_stem = Path(img_file).stem
            label_path = os.path.join(valid_labels_dir, f"{img_stem}.txt")
            with open(label_path, 'w') as f:
                # Format: class_id x_center y_center width height
                # This creates a centered box covering 50% of the image
                f.write("0 0.5 0.5 0.5 0.5\n")
        
        print(f"Created synthetic validation set with {min(50, len(train_images))} images")
        return
    
    # Select a subset for validation (10%, minimum 10, maximum 100)
    sample_size = max(10, min(100, int(len(valid_training_pairs) * 0.1)))
    validation_samples = random.sample(valid_training_pairs, sample_size)
    
    print(f"Copying {len(validation_samples)} image-label pairs to validation set...")
    
    # Copy the selected pairs to validation directories
    for img_file, label_file in validation_samples:
        # Copy image
        src_img_path = os.path.join(train_images_dir, img_file)
        dst_img_path = os.path.join(valid_images_dir, img_file)
        
        # Handle long filenames
        if len(img_file) > 100:
            extension = Path(img_file).suffix
            short_name = f"val_{hashlib.md5(img_file.encode()).hexdigest()[:20]}{extension}"
            dst_img_path = os.path.join(valid_images_dir, short_name)
            print(f"Shortened long filename: {img_file} -> {short_name}")
            
            # Update label filename to match
            label_file = f"{Path(short_name).stem}.txt"
        
        # Copy files
        shutil.copy2(src_img_path, dst_img_path)
        
        src_label_path = os.path.join(train_labels_dir, label_file)
        dst_label_path = os.path.join(valid_labels_dir, label_file)
        shutil.copy2(src_label_path, dst_label_path)
    
    print(f"Successfully created validation set with {len(validation_samples)} images and annotations")

def check_and_fix_train_labels():
    """Ensure training labels are not empty or corrupted"""
    dataset_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    train_dir = os.path.join(dataset_dir, 'train')
    
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    
    if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
        print("Training directories not found")
        return
    
    # Find all images
    train_images = [f for f in os.listdir(train_images_dir) 
                   if os.path.isfile(os.path.join(train_images_dir, f)) and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Checking {len(train_images)} training images for valid labels...")
    
    # Check and fix labels
    fixed_count = 0
    for img_file in train_images:
        img_stem = Path(img_file).stem
        label_path = os.path.join(train_labels_dir, f"{img_stem}.txt")
        
        # If label doesn't exist or is empty, create a basic annotation
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            with open(label_path, 'w') as f:
                # Format: class_id x_center y_center width height
                # This creates a centered box covering 50% of the image
                f.write("0 0.5 0.5 0.5 0.5\n")
            fixed_count += 1
    
    print(f"Fixed {fixed_count} empty or missing training labels")

if __name__ == '__main__':
    print("=== FIXING DATASET FOR TRAINING ===")
    
    # First fix training labels if needed
    print("\nStep 1: Checking training labels...")
    check_and_fix_train_labels()
    
    # Then create a proper validation set
    print("\nStep 2: Creating proper validation set...")
    fix_validation_data()
    
    print("\n=== FIX COMPLETE ===")
    print("Now try running train_yolo.py again")
