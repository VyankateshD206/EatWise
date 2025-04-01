import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import pandas as pd
import warnings
from tqdm import tqdm  # Add this import for progress bar
import json  # Add this line for JSON handling
import os
from PIL import Image


# Custom Dataset Class
class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        base_path = os.path.dirname(os.path.abspath(__file__))
        annotations_path = os.path.join(base_path, annotations_file)
        with open(annotations_path) as f:
            coco_data = json.load(f)
        self.annotations = coco_data['annotations']
        self.images = coco_data['images']

        self.img_dir = os.path.join(base_path, img_dir)
        self.transform = transform

        # Create a mapping from string labels to integer labels
        self.label_map = {label: idx for idx, label in enumerate(self.annotations['label'].unique())}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx]['file_name'])
        
        # Check if the image file exists
        if not os.path.exists(img_name):
            warnings.warn(f"Image file not found: {img_name}. Skipping this entry.")
            return None, None  # Return None for both image and target to skip this entry

        image = F.to_tensor(Image.open(img_name).convert("RGB"))

        # Extract bounding box coordinates and label
        annotation = self.annotations[idx]
        x_min = annotation['bbox'][0]
        y_min = annotation['bbox'][1]
        x_max = x_min + annotation['bbox'][2]
        y_max = y_min + annotation['bbox'][3]
        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)

        label_str = annotation['category_id']  # Assuming category_id is used for labels
        label = torch.tensor([self.label_map[label_str]], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": label.flatten()  # Ensure labels is 1D
        }

        if self.transform:
            image = self.transform(image)

        return image, target

# Load datasets
train_dataset = FoodDataset('annotations_train.json', 'train/')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

# Define the model
backbone = resnet_fpn_backbone('resnet50', weights='IMAGENET1K_V1')
num_classes = len(train_dataset.label_map) + 1
model = FasterRCNN(backbone, num_classes=num_classes)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):

        # Filter out None values
        images = [img for img in images if img is not None]
        targets = [t for t in targets if t is not None]

        # Check if there are any valid images and targets
        if len(images) == 0 or len(targets) == 0:
            continue  # Skip this iteration if no valid images or targets

        # Convert images to a list if it's not already
        if not isinstance(images, list):
            images = list(images)

        # Move images and targets to the correct device (CPU or GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Ensure that the length of images and targets match
        assert len(images) == len(targets), "Images and targets must have the same length"

        # Pass images and targets to the model
        model.to(device)  # Move the model to the device
        optimizer.zero_grad()
        losses = model(images, targets)

        # Check if losses is a dictionary or a single loss
        if isinstance(losses, dict):
            loss = sum(loss for loss in losses.values())
        else:
            loss = losses

        loss.backward()
        optimizer.step()

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("Training completed and model weights saved.")
