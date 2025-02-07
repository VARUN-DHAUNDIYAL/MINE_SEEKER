# scripts/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import RFDataset
from models.rf_landmine_model import RFClassifier  # Rename your model file as needed
from config import CONFIG
import yaml

def train_model():
    device = torch.device(CONFIG["DEVICE"])
    dataset = RFDataset(CONFIG["DATA_PROCESSED_DIR"])  # Assumes processed .npy files
    dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    model = RFClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])

    for epoch in range(CONFIG["EPOCHS"]):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}] Average Loss: {running_loss/len(dataloader):.4f}")
    
    os.makedirs(os.path.dirname(CONFIG["MODEL_SAVE_PATH"]), exist_ok=True)
    torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
    print("[INFO] Model training complete and saved!")

if __name__ == "__main__":
    train_model()
