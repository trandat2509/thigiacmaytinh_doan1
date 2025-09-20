from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms, models
from torch import nn
import torch
from PIL import Image
import io

# ===== Hàm load model =====
def load_model(path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# ===== Hàm predict =====
def predict_image(model, device, img_bytes, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred = torch.argmax(outputs, dim=1)
    return class_names[pred.item()]

# ===== Hàm tạo app =====
def create_app(model, device, class_names):
    app = FastAPI(title="Garbage Classification API")

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            img_bytes = await file.read()
            label = predict_image(model, device, img_bytes, class_names)
            return JSONResponse({"label": label})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    return app

# ===== Tự động lấy class_names từ thư mục =====
data_path = Path("garbage_classification")
class_names = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
print("Detected classes:", class_names)

# ===== Load model =====
model, device = load_model("garbage_model.pth", num_classes=len(class_names))

# ===== Tạo app ở cấp module =====
app = create_app(model, device, class_names)

# ===== Nếu muốn chạy trực tiếp python api.py =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
