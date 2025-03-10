import json
import pandas as pd

# Load COCO JSON file
with open('C:/Users/honpa/OneDrive/Desktop/Programming/DC(sem 4)/Research papers/NutrifyAI using EDAMAM implementation/valid/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Extract images, annotations, and categories
images = {img['id']: img['file_name'] for img in coco_data['images']}
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Create a list to store annotation data with food IDs and names
data = []

# Extract relevant annotation details
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']  # [x, y, width, height]
    
    # Append extracted data to the list, including food IDs and names
    data.append({
        'image_name': images[image_id],
        'category': categories[category_id],
        'food_id': category_id,  # Add food ID
        'food_name': categories[category_id]  # Add food name
    })

    # Convert extracted data into a DataFrame
df = pd.DataFrame(data)

calorie_dict = {
    'OnionPakoda': 315,        # per 100g
    'GreenChutney': 40,        # per 2 tbsp
    'Kulfi': 190,              # per serving
    'PalakPaneer': 240,        # per serving
    'Samosa': 262,             # per piece
    'Ghevar': 400,             # per serving
    'Kheer': 300,              # per serving
    'Bhatura': 300,            # per piece
    'Chole': 350,              # per serving
    'Jalebi': 150,             # per piece
    'AlooMasala': 200,         # per serving
    'ShahiPaneer': 400,        # per serving
    'Chai': 105,               # per cup
    'BhindiMasala': 150,       # per serving
    'Idli': 58,                # per piece
    'MuttonCurry': 500,        # per serving
    'RajmaCurry': 240,         # per serving
    'RasMalai': 250,           # per piece
    'Dal': 130,                # per serving
    'FishCurry': 300,          # per serving
    'WhiteRice': 205,          # per cup cooked
    'Dosa': 168,               # per piece
    'Biryani': 450,            # per serving
    'Poha': 250,               # per serving
    'Lassi': 220,              # per glass
    'AlooGobi': 150,           # per serving
    'DumAloo': 300,            # per serving
    'CoconutChutney': 100,     # per 2 tbsp
    'Kebab': 200,              # per piece
    'GulabJamun': 150          # per piece
}

# Convert DataFrame to Excel file
df.to_excel('annotations_valid.xlsx', index=False)

df = pd.read_excel('annotations_valid.xlsx')
df['CaloriesPerServing'] = df['food_name'].map(calorie_dict)
df.to_excel('annotations_valid.xlsx', index=False)

print("Calorie mapping completed and saved to 'annotations_valid.xlsx'.")