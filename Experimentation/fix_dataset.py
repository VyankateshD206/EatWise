import os
import shutil
from pathlib import Path
import re

def clean_filename(filename):
    """Clean up problematic characters in filenames"""
    # Replace certain patterns that appear to be causing issues
    # For instance, correcting "iindian" to "indian" in filenames
    cleaned = filename.replace("iindian", "indian")
    cleaned = cleaned.replace("iss-prepared", "is-prepared")
    
    return cleaned

def find_closest_match(image_name, available_images):
    """Find the closest matching image for a label file"""
    # Simplistic matching - find an image with the most similar name
    best_match = None
    best_score = 0
    
    # Get the stem (filename without extension) for comparison
    stem = Path(image_name).stem
    
    for img in available_images:
        img_stem = Path(img).stem
        
        # Calculate simple similarity score by counting common characters at the start
        i = 0
        while i < min(len(stem), len(img_stem)) and stem[i] == img_stem[i]:
            i += 1
        
        score = i
        
        if score > best_score:
            best_score = score
            best_match = img
    
    # Only return a match if it's reasonably close (at least 50% match)
    if best_score > min(len(stem), len(img_stem)) * 0.5:
        return best_match
    return None

def synchronize_labels_with_images(dataset_dir):
    """Make sure label files match their corresponding image files"""
    total_fixed = 0
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} - directory not found")
            continue
        
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"Skipping {split} - images directory not found")
            continue
        
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir, exist_ok=True)
            print(f"Created missing labels directory for {split}")
        
        # Get list of images and labels
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
        
        print(f"\nProcessing {split} split:")
        print(f"  Found {len(image_files)} images and {len(label_files)} label files")
        
        # Create a mapping of stems (filenames without extensions)
        image_stems = {Path(img).stem: img for img in image_files}
        
        # First, check for label files without corresponding images
        fixed_count = 0
        for label_file in label_files:
            label_stem = Path(label_file).stem
            
            # Check if there's a direct match
            if label_stem in image_stems:
                continue
            
            # No direct match, try to find the closest match
            matched_image = find_closest_match(label_stem, image_files)
            
            if matched_image:
                matched_stem = Path(matched_image).stem
                new_label_file = f"{matched_stem}.txt"
                
                # Rename the label file to match the image
                old_path = os.path.join(labels_dir, label_file)
                new_path = os.path.join(labels_dir, new_label_file)
                
                print(f"  Renaming label: {label_file} -> {new_label_file}")
                
                if os.path.exists(new_path):
                    # If the destination already exists, merge the content
                    with open(old_path, 'r') as f_old:
                        old_content = f_old.read()
                    
                    with open(new_path, 'a') as f_new:
                        f_new.write('\n')
                        f_new.write(old_content)
                    
                    os.remove(old_path)
                else:
                    # Simple rename
                    os.rename(old_path, new_path)
                
                fixed_count += 1
        
        # Now check for images without label files
        for image_file in image_files:
            image_stem = Path(image_file).stem
            
            # Handle extremely long filenames
            if len(image_stem) > 100:
                # Create a shortened hash-based version of the filename to avoid path length issues
                import hashlib
                short_name = hashlib.md5(image_stem.encode()).hexdigest()[:20]
                label_path = os.path.join(labels_dir, f"{short_name}.txt")
                print(f"  Using shortened name for long filename: {image_file} -> {short_name}.txt")
                
                # Save the mapping for reference
                mapping_file = os.path.join(labels_dir, "filename_mapping.txt")
                with open(mapping_file, 'a') as f:
                    f.write(f"{short_name}.txt,{image_stem}.jpg\n")
            else:
                label_path = os.path.join(labels_dir, f"{image_stem}.txt")
            
            if not os.path.exists(label_path):
                try:
                    # Create an empty label file if there's no corrupted filename to match
                    with open(label_path, 'w') as f:
                        # Empty file indicates no objects in image
                        pass
                    fixed_count += 1
                    print(f"  Created empty label for: {image_file}")
                except (OSError, IOError) as e:
                    # Handle cases where the path is too long or other issues
                    print(f"  Error creating label for {image_file}: {str(e)}")
                    # Try with a shorter name instead
                    import hashlib
                    short_name = hashlib.md5(image_stem.encode()).hexdigest()[:20]
                    fallback_path = os.path.join(labels_dir, f"{short_name}.txt")
                    with open(fallback_path, 'w') as f:
                        pass
                    fixed_count += 1
                    print(f"  Created empty label with shortened name: {short_name}.txt")
                    
                    # Keep track of the mapping
                    mapping_file = os.path.join(labels_dir, "filename_mapping.txt")
                    with open(mapping_file, 'a') as f:
                        f.write(f"{short_name}.txt,{image_stem}.jpg\n")
        
        # Remove the cache file so it will be regenerated
        cache_file = os.path.join(labels_dir, '../labels.cache')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"  Removed labels cache file for {split}")
        
        # Report results
        print(f"  Fixed {fixed_count} label files for {split} split")
        total_fixed += fixed_count
    
    return total_fixed

def ensure_clean_filenames(dataset_dir):
    """Fix problematic filenames in the dataset"""
    total_renamed = 0
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        images_dir = os.path.join(split_dir, 'images')
        if not os.path.exists(images_dir):
            continue
        
        # Get list of images
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        renamed_count = 0
        for image_file in image_files:
            cleaned_name = clean_filename(image_file)
            
            if cleaned_name != image_file:
                old_path = os.path.join(images_dir, image_file)
                new_path = os.path.join(images_dir, cleaned_name)
                
                # Rename the image file
                print(f"  Renaming image: {image_file} -> {cleaned_name}")
                os.rename(old_path, new_path)
                
                # Also rename the corresponding label file if it exists
                labels_dir = os.path.join(split_dir, 'labels')
                if os.path.exists(labels_dir):
                    old_label = f"{Path(image_file).stem}.txt"
                    new_label = f"{Path(cleaned_name).stem}.txt"
                    old_label_path = os.path.join(labels_dir, old_label)
                    new_label_path = os.path.join(labels_dir, new_label)
                    
                    if os.path.exists(old_label_path):
                        os.rename(old_label_path, new_label_path)
                        print(f"  Renaming label: {old_label} -> {new_label}")
                
                renamed_count += 1
        
        if renamed_count > 0:
            print(f"  Renamed {renamed_count} files in {split} split")
            total_renamed += renamed_count
    
    return total_renamed

def rename_problematic_images(dataset_dir):
    """Rename images with problematic filenames (e.g., too long)"""
    total_renamed = 0
    max_length = 100  # Max filename length to consider safe
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        images_dir = os.path.join(split_dir, 'images')
        if not os.path.exists(images_dir):
            continue
        
        # Get list of images
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process images with overly long filenames
        renamed_count = 0
        for image_file in image_files:
            image_stem = Path(image_file).stem
            
            if len(image_stem) > max_length:
                # Create a shortened hash-based name
                import hashlib
                extension = Path(image_file).suffix
                short_name = f"img_{hashlib.md5(image_stem.encode()).hexdigest()[:20]}{extension}"
                
                try:
                    old_path = os.path.join(images_dir, image_file)
                    new_path = os.path.join(images_dir, short_name)
                    
                    # Rename the image file
                    shutil.copy2(old_path, new_path)  # Copy first to be safe
                    os.remove(old_path)  # Then remove the original
                    
                    print(f"  Renamed long image: {image_file} -> {short_name}")
                    
                    # Keep track of the mapping
                    mapping_file = os.path.join(images_dir, "filename_mapping.txt")
                    with open(mapping_file, 'a') as f:
                        f.write(f"{short_name},{image_file}\n")
                    
                    renamed_count += 1
                    
                except (OSError, IOError) as e:
                    print(f"  Error renaming {image_file}: {str(e)}")
        
        if renamed_count > 0:
            print(f"  Renamed {renamed_count} images with long filenames in {split}")
            total_renamed += renamed_count
    
    return total_renamed

def check_image_label_alignment(dataset_dir):
    """Verify that each image has a corresponding label file"""
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue
        
        # Get list of images and labels
        image_files = [Path(f).stem for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [Path(f).stem for f in os.listdir(labels_dir) 
                      if f.lower().endswith('.txt')]
        
        # Check alignment
        images_without_labels = set(image_files) - set(label_files)
        labels_without_images = set(label_files) - set(image_files)
        
        print(f"\nAlignment check for {split} split:")
        print(f"  Images: {len(image_files)}, Labels: {len(label_files)}")
        print(f"  Images without labels: {len(images_without_labels)}")
        print(f"  Labels without images: {len(labels_without_images)}")
        
        if len(images_without_labels) > 0:
            print(f"  First few images without labels: {list(images_without_labels)[:5]}")
        
        if len(labels_without_images) > 0:
            print(f"  First few labels without images: {list(labels_without_images)[:5]}")

def main():
    dataset_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    
    print("========== FIXING DATASET ISSUES ==========")
    
    # First, rename images with problematic filenames
    print("\nStep 1: Handling excessively long filenames...")
    total_long_renamed = rename_problematic_images(dataset_dir)
    print(f"Total images with shortened filenames: {total_long_renamed}")
    
    # Next, fix other problematic filenames
    print("\nStep 2: Fixing problematic filename patterns...")
    total_renamed = ensure_clean_filenames(dataset_dir)
    print(f"Total renamed files: {total_renamed}")
    
    # Then, synchronize label files with images
    print("\nStep 3: Synchronizing label files with images...")
    total_fixed = synchronize_labels_with_images(dataset_dir)
    print(f"Total fixed label files: {total_fixed}")
    
    # Finally, verify alignment
    print("\nStep 4: Verifying image-label alignment...")
    check_image_label_alignment(dataset_dir)
    
    print("\n========== DATASET FIX COMPLETE ==========")
    print("Next steps:")
    print("1. Run python train_yolo.py to train your model")
    print("2. If issues persist, you may need to examine your dataset manually")

if __name__ == "__main__":
    main()
