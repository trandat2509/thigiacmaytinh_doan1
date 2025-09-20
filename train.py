from pathlib import Path
from collections import Counter
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File
from torchvision import datasets, transforms, models
from tqdm import tqdm
from PIL import Image
from torch import nn
import torch
import io

# ===== 1) Khám phá dataset =====
def explore_dataset(data_dir: Path):
    classes = [p.name for p in data_dir.iterdir() if p.is_dir()]
    counts = {c: len(list((data_dir / c).glob("*.*"))) for c in classes}
    return classes, counts

# ===== 2) Chuẩn bị dataloader =====
def create_dataloaders(data_dir: Path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.classes

# ===== 3) Huấn luyện model =====
def train_model(loader, num_classes, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={running_loss/len(loader):.4f}")
    return model

# ===== 4) Lưu & nạp model =====
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# ===== 5) Dự đoán ảnh đơn =====
def predict_image(model, img_bytes, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = outputs.max(1)
    return class_names[pred.item()]

# ===== 6) Triển khai FastAPI =====
def create_app(model, class_names):
    app = FastAPI()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        img_bytes = await file.read()
        label = predict_image(model, img_bytes, class_names)
        return JSONResponse({"label": label})

    return app

# ======== MAIN PIPELINE ========
if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())  # <== Dòng kiểm tra

    data_path = Path("garbage_classification") # data

    # 1) Khám phá dataset
    classes, counts = explore_dataset(data_path)
    print("Classes:", classes)
    print("Counts:", counts)

    # 2–3) Loader + train
    loader, class_names = create_dataloaders(data_path)
    model = train_model(loader, num_classes=len(class_names))

    # 4) Lưu model
    save_model(model, "garbage_model.pth")

    # 5–6) Deploy
    app = create_app(model, class_names)
    # chạy: uvicorn train_and_deploy:app --reload

    print(torch.cuda.is_available())
