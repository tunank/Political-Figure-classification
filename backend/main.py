import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models
import io

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the model
model_path = "../model/best_model.pth"
model = models.resnet18(pretrained=True)
num_classes = 5
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to fit ResNet input
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # Normalize pixel values to between -1 and 1
])

# define labels
label_dict = {0: 'Donald Trump', 1: 'Joe Biden', 2: 'Justin Trudeau', 3: 'Vladimir Putin', 4: 'Xi Jinping'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)[0]

        print("Model Raw Outputs:", outputs.tolist())  # Debugging
        print("Softmax Probabilities:", probabilities.tolist())  # Debugging

        # If probabilities contain NaN, print error and return a failure message
        if torch.isnan(probabilities).any():
            print("Error: Model output contains NaN values!")
            return {"error": "Model output contains NaN values. Check input preprocessing and model loading."}

        results = {label_dict[i]: float(probabilities[i] * 100) for i in range(num_classes)}
        sorted_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)}

        return {
            "predictions": sorted_results
        }


