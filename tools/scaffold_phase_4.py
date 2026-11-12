"""
Scaffold Generator for Phase 4: Deep Learning & Computer Vision (Weeks 11 & 12)
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_notebook(cells, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[OK] Generated Notebook: {os.path.relpath(filepath, BASE_DIR)}")

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.strip().split("\n")]}

def write_file(rel_path, content):
    full_path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Written File: {rel_path}")

# ==============================================================================
# WEEK 11: DEEP LEARNING FOUNDATIONS & ANN
# ==============================================================================
def scaffold_week_11():
    np.random.seed(42)
    n = 400
    temp = np.random.normal(70, 15, size=n)
    vibration = np.random.normal(1.2, 0.4, size=n)
    pressure = np.random.normal(30, 5, size=n)
    voltage = np.random.normal(220, 10, size=n)
    
    # Non-linear fault condition
    fault_prob = 1 / (1 + np.exp(-(-6.0 + 0.05 * temp + 2.5 * vibration + 0.1 * pressure)))
    fault = (np.random.rand(n) < fault_prob).astype(int)
    
    df = pd.DataFrame({
        "Temperature_C": np.round(temp, 1),
        "Vibration_mm_s": np.round(vibration, 2),
        "Pressure_PSI": np.round(pressure, 1),
        "Voltage_V": np.round(voltage, 1),
        "Fault_Detected": fault
    })
    csv_path = os.path.join(BASE_DIR, "phase-4-deep-learning-computer-vision/week-11-deep-learning-ann/datasets/sensor_fault_detection.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated Week 11 dataset: {df.shape}")

    write_file("phase-4-deep-learning-computer-vision/week-11-deep-learning-ann/README.md", """# Week 11 — Deep Learning Foundations & Artificial Neural Networks (ANN)
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand the mathematical mechanics of Artificial Neurons, Perceptrons, and MLPs.
2. Master Activation Functions (ReLU, Sigmoid, Softmax) and Loss Functions (Cross-Entropy).
3. Build a Multi-Layer Perceptron using **PyTorch**.
4. Implement a custom Training Loop with Forward Propagation, Backpropagation, and Adam Optimization.
5. Track and plot Training vs Validation Loss and Accuracy curves to prevent overfitting.
""")

    nb_path = os.path.join(BASE_DIR, "phase-4-deep-learning-computer-vision/week-11-deep-learning-ann/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 11 — Deep Learning Foundations & ANN
### Building and Training an Artificial Neural Network in PyTorch
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | PyTorch version: {torch.__version__}")"""),
        md_cell("""## 1. Data Ingestion & Tensor Conversion"""),
        code_cell("""df = pd.read_csv("datasets/sensor_fault_detection.csv")
X = df.drop(columns=["Fault_Detected"]).values
y = df["Fault_Detected"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train).unsqueeze(1))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)"""),
        md_cell("""## 2. Define the Neural Network Architecture (MLP)"""),
        code_cell("""class SensorFaultANN(nn.Module):
    def __init__(self, input_dim):
        super(SensorFaultANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SensorFaultANN(input_dim=X_train.shape[1]).to(device)
print(model)"""),
        md_cell("""## 3. Training Loop with Loss Tracking"""),
        code_cell("""criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 40
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    
    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx)
            l = criterion(preds, by)
            val_loss += l.item() * bx.size(0)
            
    epoch_val_loss = val_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")"""),
        md_cell("""## 4. Plot Loss Curves"""),
        code_cell("""plt.figure(figsize=(9, 5))
plt.plot(train_losses, label="Training Loss", color="royalblue")
plt.plot(val_losses, label="Validation Loss", color="coral", linestyle="--")
plt.title("ANN Training vs Validation Loss Curve", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.show()""")
    ]
    make_notebook(cells, nb_path)

# ==============================================================================
# WEEK 12: COMPUTER VISION & CNN
# ==============================================================================
def scaffold_week_12():
    write_file("phase-4-deep-learning-computer-vision/week-12-computer-vision-cnn/README.md", """# Week 12 — Computer Vision & Convolutional Neural Networks (CNN)
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand Digital Image Representation (Pixels, RGB Channels, Tensors).
2. Learn Convolutional Filters, Strides, Padding, and Pooling layers.
3. Architect and train a Convolutional Neural Network (CNN) in PyTorch.
4. Implement Data Augmentation (random flips, rotations) to prevent visual overfitting.
5. Understand the principles of Transfer Learning (ResNet / MobileNet backbones).
""")

    nb_path = os.path.join(BASE_DIR, "phase-4-deep-learning-computer-vision/week-12-computer-vision-cnn/starter_notebook.ipynb")
    cells = [
        md_cell("""# Week 12 — Computer Vision & Convolutional Neural Networks (CNN)
### End-to-End Image Classification Pipeline
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship**"""),
        code_cell("""import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")"""),
        md_cell("""## 1. Image Data Augmentation & Loading (Fashion-MNIST)"""),
        code_cell("""transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f"Loaded {len(train_set)} training images across {len(classes)} classes.")"""),
        md_cell("""## 2. Convolutional Neural Network Architecture"""),
        code_cell("""class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 28x28 -> 14x14
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = FashionCNN().to(device)
print(model)"""),
        md_cell("""## 3. Training Loop & Validation Evaluation
*(Interns: Execute training for 5-10 epochs and report test accuracy and confusion matrix)*""")
    ]
    make_notebook(cells, nb_path)

if __name__ == "__main__":
    print("Scaffolding Phase 4...")
    scaffold_week_11()
    scaffold_week_12()
    print("Phase 4 complete!")
