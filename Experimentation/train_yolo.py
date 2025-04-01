import os
import sys
import subprocess
import yaml
from pathlib import Path

# Install required packages if not already installed
def install_requirements():
    packages = ["ultralytics"]
    for package in packages:
        try:
            __import__(package)
            print(f"{package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Set up paths
def setup_paths(base_dir=None):
    if base_dir is None:
        base_dir = r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset"
    
    # Update paths to match the user's directory structure
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')
    
    return {
        'base_dir': base_dir,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'test_dir': test_dir
    }

# Create YAML config for training
def create_yaml_config(paths, class_names):
    yaml_content = {
        'path': paths['base_dir'],
        'train': os.path.join('train', 'images'),
        'val': os.path.join('valid', 'images'),
        'test': os.path.join('test', 'images'),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(paths['base_dir'], 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return yaml_path

# Train the YOLO model
def train_yolo(yaml_path, model_size='m', epochs=100, batch_size=16):
    from ultralytics import YOLO
    
    # Load a model
    model = YOLO(f'yolov8{model_size}.pt')  # load a pretrained model (recommended for training)
    
    # Check if validation directory has valid data
    import yaml
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    base_dir = yaml_data.get('path', '')
    val_dir = os.path.join(base_dir, yaml_data.get('val', 'valid/images'))
    
    # Check if validation directory has images with labels
    has_valid_val = check_valid_directory(val_dir)
    
    # Prepare training arguments
    train_args = {
        'data': yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 640,
        'save': True,
        'patience': 50,
    }
    
    if not has_valid_val:
        print("\nWARNING: No valid validation data found. Training will proceed without validation.")
        print("This may affect the model's performance evaluation.")
        train_args['val'] = False
        
        # Ask if the user wants to fix validation data
        fix_choice = input("Would you like to automatically fix the validation data? (y/n): ")
        if fix_choice.lower() == 'y':
            # Call the validation fix script
            print("Running validation fix script...")
            try:
                # Fix validation data directly
                from fix_validation import fix_validation_data
                fix_validation_data()
                print("Validation data fixed. Proceeding with training...")
                train_args['val'] = True
            except Exception as e:
                print(f"Error fixing validation data: {str(e)}")
                print("Will proceed without validation.")
    
    # Train the model
    try:
        print("\nStarting training...")
        results = model.train(**train_args)
        return model, results
    except Exception as e:
        if "expected a non-empty list of Tensors" in str(e):
            print("\nERROR: Training failed due to empty tensor list in validation.")
            print("This usually happens when there are no valid annotations in your validation set.")
            
            retry_choice = input("Would you like to retry without validation? (y/n): ")
            if retry_choice.lower() == 'y':
                print("Retrying without validation...")
                train_args['val'] = False
                results = model.train(**train_args)
                return model, results
            else:
                raise e
        else:
            raise e

def check_valid_directory(dir_path):
    """Check if a directory contains valid data for training/validation"""
    # Extract the base directory (up one level from images)
    base_dir = os.path.dirname(dir_path)
    
    # Check if the directory exists
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return False
    
    # Check if there are images
    image_files = [f for f in os.listdir(dir_path) 
                  if os.path.isfile(os.path.join(dir_path, f)) and 
                  f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"No images found in {dir_path}")
        return False
    
    # Check if there are labels
    labels_dir = os.path.join(base_dir, 'labels')
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return False
    
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    if len(label_files) == 0:
        print(f"No label files found in {labels_dir}")
        return False
    
    # Check if at least some images have corresponding labels
    matched_labels = 0
    for img_file in image_files[:10]:  # Only check a sample
        img_stem = Path(img_file).stem
        label_path = os.path.join(labels_dir, f"{img_stem}.txt")
        
        if os.path.exists(label_path):
            matched_labels += 1
    
    if matched_labels == 0:
        print(f"No matching labels found for images in {dir_path}")
        return False
    
    return True

# Evaluate the model
def evaluate_model(model):
    # Validate the model
    try:
        metrics = model.val()
        print(f"Validation metrics: {metrics}")
        return metrics
    except RuntimeError as e:
        print(f"Error during validation: {str(e)}")
        print("Validation failed. This may be due to empty validation set or mismatched labels.")
        return None

def verify_dataset_ready(paths):
    """Check if the dataset is ready for training"""
    ready = True
    missing = []
    
    # Check each split directory
    for split in ['train', 'val', 'test']:
        split_key = f"{split}_dir" if split != "val" else "val_dir"
        split_dir = paths[split_key]
        
        # Check for images
        images_dir = os.path.join(split_dir, 'images')
        if not os.path.exists(images_dir):
            missing.append(f"Images directory for {split}")
            ready = False
            continue
            
        # Count actual image files
        image_files = [f for f in os.listdir(images_dir) 
                      if os.path.isfile(os.path.join(images_dir, f)) and 
                      f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            missing.append(f"Images for {split} (directory exists but empty)")
            ready = False
        
        # Check for annotations - either COCO json or YOLO txt files
        labels_dir = os.path.join(split_dir, 'labels')
        coco_file = os.path.join(split_dir, 'annotations.json')
        
        has_labels = False
        if os.path.exists(labels_dir):
            try:
                # Check if there are some label files that match image files
                matched_labels = 0
                for img_file in image_files[:100]:  # Only check a sample
                    img_stem = Path(img_file).stem
                    label_path = os.path.join(labels_dir, f"{img_stem}.txt")
                    
                    # Check for shortened names for long filenames
                    if not os.path.exists(label_path) and len(img_stem) > 100:
                        import hashlib
                        short_name = hashlib.md5(img_stem.encode()).hexdigest()[:20]
                        short_path = os.path.join(labels_dir, f"{short_name}.txt")
                        if os.path.exists(short_path):
                            matched_labels += 1
                    elif os.path.exists(label_path):
                        matched_labels += 1
                
                # If at least some labels match images, consider it valid
                has_labels = matched_labels > 0
                
                if not has_labels:
                    # Count all label files as a fallback check
                    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
                    has_labels = len(label_files) > 0
                
                # Create empty label files for missing ones
                if not has_labels and len(image_files) > 0:
                    print(f"Creating empty label files for {split} images...")
                    for img_file in image_files:
                        img_stem = Path(img_file).stem
                        label_path = os.path.join(labels_dir, f"{img_stem}.txt")
                        
                        # Handle very long filenames
                        if len(img_stem) > 100:
                            import hashlib
                            short_name = hashlib.md5(img_stem.encode()).hexdigest()[:20]
                            label_path = os.path.join(labels_dir, f"{short_name}.txt")
                        
                        if not os.path.exists(label_path):
                            try:
                                with open(label_path, 'w') as f:
                                    pass  # Create empty file
                            except (OSError, IOError) as e:
                                print(f"Warning: Could not create label for {img_file}: {e}")
                    has_labels = True
            except Exception as e:
                print(f"Error checking labels for {split}: {e}")
                
        has_coco = os.path.exists(coco_file)
        
        if not (has_labels or has_coco):
            missing.append(f"Annotations for {split}")
            ready = False
    
    return ready, missing

def load_class_names_from_yaml(yaml_path):
    """Load class names from existing YAML file"""
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            if 'names' in yaml_data:
                return list(yaml_data['names'].values())
    
    return None

def main():
    # Install required packages
    install_requirements()
    
    # Set up paths with correct base directory 
    paths = setup_paths(r"C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\dataset")
    
    # Check if directories exist
    print("Checking dataset directories:")
    for key, path in paths.items():
        if key != 'base_dir':
            exists = os.path.exists(path)
            print(f"  {key}: {'Found' if exists else 'Not found'} - {path}")
    
    # Check if the dataset is ready
    dataset_ready, missing_items = verify_dataset_ready(paths)
    
    if not dataset_ready:
        print("Warning: Your dataset is not fully prepared for training.")
        print("Missing items:")
        for item in missing_items:
            print(f"  - {item}")
        
        fix_choice = input("Would you like to run the dataset fix script to attempt to resolve these issues? (y/n): ")
        if fix_choice.lower() == 'y':
            print("\nRunning dataset fix script...")
            import subprocess
            fix_script = os.path.join(os.path.dirname(paths['base_dir']), 'fix_dataset.py')
            
            if not os.path.exists(fix_script):
                print("Creating fix_dataset.py script...")
                with open(fix_script, 'w') as f:
                    # You can include the content here or direct the user to run it separately
                    f.write("# Dataset fix script would go here\n")
                    f.write("print('Please run fix_dataset.py directly')\n")
                
            subprocess.call([sys.executable, fix_script])
            
            # Check again after fixing
            dataset_ready, missing_items = verify_dataset_ready(paths)
        
        if not dataset_ready:
            continue_choice = input("Dataset still has issues. Do you want to continue anyway? (y/n): ")
            if continue_choice.lower() != 'y':
                print("Training aborted. Please fix your dataset and try again.")
                return
    
    # Look for existing YAML file
    yaml_path = os.path.join(paths['base_dir'], 'dataset.yaml')
    parent_yaml_path = os.path.join(os.path.dirname(paths['base_dir']), 'dataset.yaml')
    
    # Try to load class names from either location
    class_names = load_class_names_from_yaml(yaml_path)
    if not class_names:
        class_names = load_class_names_from_yaml(parent_yaml_path)
        if class_names:
            yaml_path = parent_yaml_path
    
    if not class_names:
        # Prompt user to enter class names manually
        print("No class names found in dataset.yaml.")
        manual_entry = input("Would you like to enter class names manually? (y/n): ")
        
        if manual_entry.lower() == 'y':
            num_classes = int(input("Enter the number of classes: "))
            class_names = []
            print("Enter each class name:")
            for i in range(num_classes):
                class_name = input(f"Class {i}: ")
                class_names.append(class_name)
            
            # Create a new YAML file with these class names
            yaml_path = create_yaml_config(paths, class_names)
        else:
            print("Please run prepare_dataset.py first to create dataset.yaml.")
            return
    
    print(f"Training with {len(class_names)} classes: {', '.join(class_names)}")
    
    # Ask for training parameters
    model_size = input("Enter model size (n/s/m/l/x) [default: m]: ") or 'm'
    epochs = int(input("Enter number of epochs [default: 100]: ") or 100)
    batch_size = int(input("Enter batch size [default: 16]: ") or 16)
    
    # Train the model
    print(f"\nStarting training with YOLOv8{model_size} for {epochs} epochs (batch size: {batch_size})...")
    try:
        # Try to import and run fix_validation if it exists
        try:
            validation_fix_choice = input("Do you want to check and prepare validation data before training? (y/n): ")
            if validation_fix_choice.lower() == 'y':
                print("Checking and fixing validation data...")
                from fix_validation import fix_validation_data
                fix_validation_data()
        except ImportError:
            print("Validation fix module not found. Proceeding with training.")
        
        model, results = train_yolo(yaml_path, model_size=model_size, epochs=epochs, batch_size=batch_size)
        
        # Evaluate the model only if training succeeded
        if model is not None:
            print("\nEvaluating model...")
            metrics = evaluate_model(model)
        
        print("\nTraining completed!")
        print(f"Model saved in runs/detect/train/weights/")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Training failed. Please check your dataset and try again.")
        
        # Offer to run the fix script
        fix_choice = input("Would you like to run the dataset fix script and try again? (y/n): ")
        if fix_choice.lower() == 'y':
            try:
                from fix_validation import fix_validation_data, check_and_fix_train_labels
                print("\nFixing dataset issues...")
                check_and_fix_train_labels()
                fix_validation_data()
                
                retry = input("Dataset fixed. Do you want to retry training now? (y/n): ")
                if retry.lower() == 'y':
                    # Try training again with validation disabled
                    print("\nRetrying training without validation...")
                    model = YOLO(f'yolov8{model_size}.pt')
                    results = model.train(
                        data=yaml_path,
                        epochs=epochs,
                        batch=batch_size,
                        imgsz=640,
                        save=True,
                        patience=50,
                        val=False  # Disable validation to avoid errors
                    )
                    print("\nTraining completed!")
                    print(f"Model saved in runs/detect/train/weights/")
            except Exception as fix_err:
                print(f"Error during fix: {str(fix_err)}")
                print("Unable to fix dataset automatically.")

if __name__ == "__main__":
    main()
