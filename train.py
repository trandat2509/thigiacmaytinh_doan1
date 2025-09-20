from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch import nn
import torch

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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader, dataset.classes

# ===== 3) Huấn luyện model =====
def train_model(loader, num_classes, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Sử dụng weights chuẩn ImageNet
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
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

# ===== 4) Lưu model =====
def save_model(model, path="garbage_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# ======== MAIN PIPELINE ========
if __name__ == "__main__":
    data_path = Path("garbage_classification")

    # Khám phá dataset
    classes, counts = explore_dataset(data_path)
    print("Classes:", classes)
    print("Counts:", counts)

    # Loader + train
    loader, class_names = create_dataloaders(data_path)
    model = train_model(loader, num_classes=len(class_names), epochs=3)

    # Lưu model
    save_model(model, "garbage_model.pth")
