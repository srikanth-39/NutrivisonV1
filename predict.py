import torch
from pathlib import Path
from PIL import Image

from model import build_model
from config import BEST_MODEL
from transforms import test_transform
from retrive import retrive


# -----------------------------
# Load Checkpoint
# -----------------------------
checkpoint = torch.load(
    BEST_MODEL,
    map_location="cpu"
)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

# -----------------------------
# Build Model
# -----------------------------
model = build_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image_path):
    # Ensure path is a string or Path object
    image_path = Path(image_path)
    
    if not image_path.exists():
        return {"error": f"Path not found: {image_path}"}

    image = Image.open(image_path).convert("RGB")
    image = test_transform(image).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    food_name = class_names[predicted.item()]

    return {
        "food": food_name,
        "confidence": f"{confidence.item() * 100:.2f}%"
    }


# -----------------------------
# Test / Interactive Prompt
# -----------------------------
if __name__ == "__main__":
    user_input = input("Enter Image Path : ")
    
    # Automatically clean quotes and format the path safely
    cleaned_path = user_input.strip().strip('"').strip("'")
    
    result = predict_image(cleaned_path)
    
    # Check if prediction encountered an error
    if "error" in result:
        print(f"\n[Error] {result['error']}")
    else:
        print("\nPrediction Result:")
        print(result)
        
        food_name = result.get("food", "")
        if food_name:
            print(f"\nFetching metadata for: {food_name}...")
            data = retrive(food_name)
            
            # Safely check if data was actually returned from the table
            if data and len(data) > 0:
                print("-" * 30)
                print(f"ID           : {data[0][0]}")
                print(f"Food Item    : {data[0][1]}")
                print(f"Calories     : {data[0][2]}")
                print(f"Protein      : {data[0][3]}")
                print(f"Carbohydrates: {data[0][4]}")
                print(f"Fat          : {data[0][5]}")
                print(f"Fiber        : {data[0][6]}")
                print("-" * 30)
            else:
                print(f"-> Warning: No metadata found in the database for '{food_name}'.")