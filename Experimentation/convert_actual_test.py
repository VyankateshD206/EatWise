import json
import pandas as pd

# Load COCO JSON file
with open('C:/Users/honpa/OneDrive/Desktop/Programming/DC(sem 4)/Research papers/NutrifyAI using EDAMAM implementation/exp/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Extract images, annotations, and categories
images = {img['id']: img['file_name'] for img in coco_data['images']}
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Create a list to store annotation data
data = []

# Extract relevant annotation details
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']  # [x, y, width, height]
    
    # Append extracted data to the list
    data.append({
        'image_name': images[image_id],
        'x_min': ann['bbox'][0],  # x-coordinate of the top-left corner
        'y_min': ann['bbox'][1],  # y-coordinate of the top-left corner
        'x_max': ann['bbox'][0] + ann['bbox'][2],  # x-coordinate of the bottom-right corner
        'y_max': ann['bbox'][1] + ann['bbox'][3],  # y-coordinate of the bottom-right corner
        'label': categories[category_id]  # Class label of the object
    })

# Convert extracted data into a DataFrame
df = pd.DataFrame(data)

# Convert DataFrame to Excel file
df.to_excel('annotations_test.xlsx', index=False)

print("Validation annotations saved to 'annotations_test.xlsx'.")
