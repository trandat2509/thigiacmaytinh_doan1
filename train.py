from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns


# ===== 1) Khám phá dataset =====
def explore_dataset(data_dir: Path):
    classes = [p.name for p in data_dir.iterdir() if p.is_dir()]
    counts = {c: len(list((data_dir / c).glob("*.*"))) for c in classes}
    return classes, counts


# ===== 2) Transform cho ảnh =====
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ===== 3) Tạo DataLoader =====
def create_dataloaders(data_dir: Path, batch_size=32, val_ratio=0.2):
    transform = get_transform()
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_names


# ===== 4) Trích xuất đặc trưng bằng ResNet18 pretrained =====
def extract_features(loader, device):
    """Trả về (features, labels) dạng numpy"""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()   # bỏ layer phân lớp cuối
    model.to(device)
    model.eval()

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = model(imgs)                 # (batch, 512)
            feats.append(out.cpu().numpy())
            labels.append(lbls.numpy())
    X = np.vstack(feats)
    y = np.hstack(labels)
    return X, y


# ===== 5) Vẽ ma trận nhầm lẫn =====
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ===== 6) Huấn luyện & Đánh giá KNN =====
def train_and_evaluate_knn(X_train, y_train, X_val, y_val, class_names,
                           n_neighbors=5):
    print(f"Đang huấn luyện KNN với k={n_neighbors} ...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

    start_time = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Thời gian huấn luyện KNN: {train_time:.2f} giây")

    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    print(f"Độ chính xác Validation: {acc:.2f}%\n")

    print("=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # --- Vẽ ma trận nhầm lẫn ---
    plot_confusion_matrix(y_val, y_pred, class_names)
    return knn


# ===== MAIN =====
if __name__ == "__main__":
    data_path = Path("garbage_classification")   # Thay đường dẫn nếu cần

    # 1) Thông tin dataset
    classes, counts = explore_dataset(data_path)
    print("Classes:", classes)
    print("Counts:", counts)

    # 2) DataLoader
    train_loader, val_loader, class_names = create_dataloaders(
        data_path, batch_size=32, val_ratio=0.2
    )

    # 3) Trích xuất đặc trưng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Đang trích xuất đặc trưng bằng ResNet18 ...")
    X_train, y_train = extract_features(train_loader, device)
    X_val, y_val = extract_features(val_loader, device)
    print(f"Đặc trưng train: {X_train.shape}, val: {X_val.shape}")

    # 4) Huấn luyện & đánh giá KNN + ma trận nhầm lẫn
    knn_model = train_and_evaluate_knn(
        X_train, y_train, X_val, y_val, class_names, n_neighbors=5
    )
