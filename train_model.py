# train_test.py
import sys
import torch
from torch.utils.data import DataLoader
from dataset import LungNoduleDataset
from model import Simple3DCNN
from desnet3D import densenet3d121
from resnet import ResNet3D
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import time

def evaluate(model, test_loader, device):
    model.eval()
    preds = []
    gts = []

    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(device)
            labels = labels.to(device)

            outputs = model(patches).squeeze()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            preds.extend(predicted.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    preds = np.array(preds)
    gts = np.array(gts)

    precision = precision_score(gts, preds, zero_division=0)
    recall = recall_score(gts, preds, zero_division=0)
    f1 = f1_score(gts, preds, zero_division=0)
    acc = accuracy_score(gts, preds)

    # support
    pos_support = np.sum(gts == 1)
    neg_support = np.sum(gts == 0)

    print("\n===== TEST METRICS =====")
    print("              precision    recall     F1-score    support")
    print(f"positive     {precision:10.4f} {recall:10.4f} {f1:10.4f} {pos_support:10d}")
    print(f"negative     {(1-precision):10.4f} {(1-recall):10.4f} {(1-f1):10.4f} {neg_support:10d}")
    print(f"accuracy     {acc:10.4f}")
    print("=====================================")

    return precision, recall, f1, acc


def predict(model_path, patch):
    model = densenet3d121()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        x = torch.tensor(patch).float().unsqueeze(0)
        logit = model(x)
        prob = torch.sigmoid(logit).item()

        return {
            "probability_positive": prob,
            "predicted_label": int(prob > 0.5)
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = LungNoduleDataset(
        csv_file=r"D:\LUNA\LUNA25_Public_Training_Development_Data.csv",
        data_folder="train_data",
        patch_size=32
    )
    test_dataset = LungNoduleDataset(
        csv_file=r"D:\LUNA\LUNA25_Public_Training_Development_Data.csv",
        data_folder="test_data",
        patch_size=32
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = densenet3d121().to(device)
    # Count class balance
    pos = np.sum([train_dataset.data.iloc[i]["label"] == 1 for i in train_dataset.valid_rows])
    neg = np.sum([train_dataset.data.iloc[i]["label"] == 0 for i in train_dataset.valid_rows])

    pos_weight = torch.tensor([neg / pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for patches, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    print("Training finished!")

    # Save
    torch.save(model.state_dict(), "resnet.pth")
    print("Model saved to resnet.pth")

    # Evaluate
    evaluate(model, test_loader, device)

    print(f"Total time: {time.time()-start:.2f} seconds")


if __name__ == "__main__":
    main()