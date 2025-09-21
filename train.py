from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch import nn
import torch
from torch.utils.data import random_split, DataLoader
import time
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import numpy as np

# ===== 1) Khám phá dataset =====
def explore_dataset(data_dir: Path):
    classes = [p.name for p in data_dir.iterdir() if p.is_dir()]
    counts = {c: len(list((data_dir / c).glob("*.*"))) for c in classes}
    return classes, counts

# ===== 2) Chuẩn bị transform =====
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# ===== 3) Chuẩn bị DataLoader =====
def create_dataloaders(data_dir: Path, batch_size=32, val_ratio=0.2):
    train_transform, val_transform = get_transforms()
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = dataset.classes

    # Chia train/validation
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Update transform cho val_dataset
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, class_names

# ===== 4) Đánh giá mô hình =====
def evaluate_model(model, data_loader, criterion, device="cuda"):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ===== 4) Huấn luyện model =====
def train_model(train_loader, val_loader, num_classes, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load ResNet18 pretrained ImageNet
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Đinh nghĩa loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Tính thời gian tổng
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        # ===== Training =====
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            model.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total * 100

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "garbage_model_best.pth")
            print(f"--> Saved best model at epoch {epoch+1} with acc {best_acc:.2f}%")

        scheduler.step()

        print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Kết thúc và in tổng thời gian
    total_time = time.time() - start_time
    print(f"Tổng thời gian train {epochs} epoch: {total_time/60:.2f} phút")

    return model

# ===== 5) Đo thời gian suy luận =====
def evaluate_best_model(model_path: str,
                        model: torch.nn.Module,
                        test_loader,
                        criterion,
                        label_encoder,
                        device: torch.device):
    """
    Đánh giá mô hình đã huấn luyện:
    - Tải checkpoint model tốt nhất
    - Tính Loss & Accuracy
    - Đo thời gian inference
    - In classification report + precision/recall/F1
    """
    # Load best weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Đo thời gian inference trên toàn bộ test set
    start_time = time.time()
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    end_time = time.time()

    total_test_time = end_time - start_time
    avg_test_time_per_batch = total_test_time / len(test_loader)
    avg_test_time_per_sample = total_test_time / len(test_loader.dataset)

    print(f"\n⏱ Test inference time: {total_test_time:.4f}s "
          f"({avg_test_time_per_batch:.4f}s/batch, "
          f"{avg_test_time_per_sample*1000:.4f} ms/sample)")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Dự đoán để tạo classification report
    start_pred_time = time.time()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    end_pred_time = time.time()

    print(f"⏱ Classification report generation time: "
          f"{end_pred_time - start_pred_time:.4f}s")

    # Báo cáo chi tiết cho từng lớp
    print(classification_report(y_true, y_pred,
                                target_names=label_encoder.classes_))

    # Weighted metrics
    report = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    print(f"Weighted Precision: {report['weighted avg']['precision']*100:.2f}%")
    print(f"Weighted Recall:    {report['weighted avg']['recall']*100:.2f}%")
    print(f"Weighted F1-Score:  {report['weighted avg']['f1-score']*100:.2f}%")


# ===== 5) Lưu model =====
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
    train_loader, val_loader, class_names = create_dataloaders(data_path, batch_size=32, val_ratio=0.2)
    model = train_model(train_loader, val_loader, num_classes=len(class_names), epochs=10)

    # Lưu model
    save_model(model, "garbage_model.pth")
