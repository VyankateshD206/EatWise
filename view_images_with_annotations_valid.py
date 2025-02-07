import json
import pandas as pd

# Load annotations
with open('valid/_annotations.coco.json') as f:
    data = json.load(f)

# Calorie information mapping (example data)
calorie_info = {
    "Apple": 95,
    "Chapathi": 150,
    "Chicken Gravy": 250,
    "Fries": 365,
    "Idli": 50,
    "Pizza": 285,
    "Rice": 206,
    "Soda": 150,
    "Tomato": 22,
    "Vada": 150,
    "Banana": 105,
    "Burger": 354
}

# Function to save predictions to Excel
def save_predictions_to_excel(data):
    predictions = []
    
    for image in data['images']:
        image_id = image['id']
        file_name = image['file_name']
        
        # Collect annotations for each image
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id']
                category_name = next((cat['name'] for cat in data['categories'] if cat['id'] == category_id), "Unknown")
                
                # Save prediction
                predictions.append({
                    "Image": file_name,
                    "Food Item": category_name,
                    "Calories": calorie_info.get(category_name, "N/A")
                })

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(predictions)
    df.to_excel('food_predictions_validation.xlsx', index=False)

# Call the function
save_predictions_to_excel(data)
