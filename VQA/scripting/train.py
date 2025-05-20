import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vqa_dataset import SimpleVQADataset
from vqa_model import SimpleVQAModel

# Paths: set these to your dataset locations
base = r'D:\Projects\asfdgfhjghk\dataset'
D:\Projects\asfdgfhjghk\VQA\PKL\val_image_features.pkl
train_q   = os.path.join(base, 'v2_Questions_Train_mscoco', 'v2_OpenEnded_mscoco_train2014_questions.json')
train_a   = os.path.join(base, 'v2_Annotations_Train_mscoco', 'v2_mscoco_train2014_annotations.json')
train_img = os.path.join(base, 'train2014', 'train2014')
val_q     = os.path.join(base, 'v2_Questions_Val_mscoco',   'v2_OpenEnded_mscoco_val2014_questions.json')
val_a     = os.path.join(base, 'v2_Annotations_Val_mscoco', 'v2_mscoco_val2014_annotations.json')
val_img   = os.path.join(base, 'val2014', 'val2014')

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
MAX_Q_LEN = 20


def train():
    # Create datasets and loaders (use workers + pin_memory for faster IO)
    train_ds = SimpleVQADataset(train_q, train_a, train_img, max_q_len=MAX_Q_LEN)
    val_ds   = SimpleVQADataset(val_q,   val_a,   val_img,   max_q_len=MAX_Q_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, Loss, Optimizer
    model = SimpleVQAModel(
        vocab_size=len(train_ds.word2idx),
        num_answers=len(train_ds.idx2ans)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for imgs, qs, labels in train_loader:
            imgs, qs, labels = imgs.to(device), qs.to(device), labels.to(device)
            outputs = model(imgs, qs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_total, val_correct, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, qs, labels in val_loader:
                imgs, qs, labels = imgs.to(device), qs.to(device), labels.to(device)
                outputs = model(imgs, qs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{EPOCHS}"
              f" - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
              f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == '__main__':
    train()