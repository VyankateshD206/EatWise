# YOLO Model Training for Food Dataset

This project contains scripts to prepare your food dataset and train a YOLO object detection model using COCO format annotations.

## Dataset

The dataset contains 30 food classes: Bhatura, Chole, GulabJamun, RasMalai, Poha, Dosa, DumAloo, Lassi, OnionPakoda, WhiteRice, Biryani, Kulfi, Jalebi, AlooMasala, BhindiMasala, CoconutChutney, FishCurry, Chai, RajmaCurry, Idli, GreenChutney, MuttonCurry, PalakPaneer, Ghevar, AlooGobi, ShahiPaneer, Kheer, Dal, Kebab, Samosa

## Setup

1. Install required dependencies:
   ```
   pip install ultralytics pandas openpyxl pyyaml
   ```

2. Organize your dataset with the following structure:
   ```
   C:\Users\honpa\OneDrive\Desktop\Programming\DC(sem 4)\Experimentation\
     ├── train/
     │    ├── images/   (place training images here)
     │    └── annotations.json  (COCO format)
     ├── valid/
     │    ├── images/   (place validation images here)
     │    └── annotations.json  (COCO format)
     └── test/
          ├── images/   (place test images here)
          └── annotations.json  (COCO format)
   ```

## Complete Workflow

### 1. Create Dataset Structure

Run the preparation script to create the necessary directories:

```
python prepare_dataset.py
```

When prompted, provide the path to your Excel annotations file (e.g., `annotations_test.xlsx`). The script will create the dataset structure if it doesn't exist.

### 2. Convert Excel Annotations to COCO Format

If your annotations are in Excel format, convert them to COCO format:

```
python excel_to_coco.py
```

This script will:
- Read your Excel file with annotations
- Ask you to specify which dataset split to process
- Create COCO format annotations in the appropriate directories

### 3. (Optional) Convert COCO to YOLO Format

If you need YOLO format annotations rather than using COCO directly:

```
python coco_to_yolo.py
```

This script will convert your COCO annotations to YOLO format text files in the 'labels' directory.

### 4. Train the Model

Run the training script:

```
python train_yolo.py
```

The script will:
- Verify your dataset is ready
- Ask for training parameters (model size, epochs, batch size)
- Train YOLOv8 on your dataset
- Evaluate the model on your validation set

### 5. Results

After training, you'll find:
- Trained model weights in 'runs/detect/train/weights/'
- Training metrics and plots in the 'runs/detect/train/' directory

## Troubleshooting

If you encounter issues:

1. Ensure your images are correctly placed in the respective 'images' directories
2. Check that annotations are properly formatted
3. Run `python prepare_dataset.py` again to verify the dataset structure
4. Look for error messages which may indicate specific issues
