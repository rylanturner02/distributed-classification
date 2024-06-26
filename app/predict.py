import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import json
import io
from PIL import Image
from torchvision import transforms
from app.model import TreeClassifier

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = TreeClassifier()
    
    if not os.path.exists('tree_classifier.pth'):
        print("ERROR: Model file 'tree_classifier.pth' not found. Train the model first!")
        raise FileNotFoundError("Model file not found")
    
    model.load_state_dict(torch.load('tree_classifier.pth', map_location='cpu'))
    model.eval()

    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except Exception as e:
    print(f"Error initializing model: {e}")
    raise

# Setup transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Image transformation
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return {
            "class_name": class_names[str(predicted.item())],
            "confidence": float(confidence.item()),
            "class_id": int(predicted.item())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
