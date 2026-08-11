import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from model import build_model
from config import BEST_MODEL
from transforms import test_transform
from retrive import retrive

app = FastAPI(title="NutriVision API", version="1.0")

# Load model globally on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(BEST_MODEL, map_location=device)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

# Loads your current MobileNetV3 Small model structure
model = build_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

@app.post("/predict")
async def predict_food_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        tensor_image = test_transform(image).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            outputs = model(tensor_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            
        food_name = class_names[predicted.item()]
        conf_score = round(confidence.item() * 100, 2)
        
        # Fetch metadata from your database table
        db_data = retrive(food_name)
        
        nutrition_info = {}
        if db_data and len(db_data) > 0:
            nutrition_info = {
                "id": db_data[0][0],
                "food_item": db_data[0][1],
                "calories": db_data[0][2],
                "protein": db_data[0][3],
                "carbohydrates": db_data[0][4],
                "fat": db_data[0][5],
                "fiber": db_data[0][6]
            }
        else:
            nutrition_info = {"message": "Metadata not found in database."}
            
        return {
            "success": True,
            "prediction": food_name,
            "confidence": f"{conf_score}%",
            "nutrition": nutrition_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)