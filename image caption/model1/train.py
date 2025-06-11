import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except ImportError:
    print("Warning: pycocoevalcap not installed. CIDEr metric will be skipped.")
    CIDER_AVAILABLE = False

from model import EncoderCNN, DecoderWithAttention
from dataset import Flickr8kDataset, collate_fn

# Paths
CAPTIONS_PATH = r"D:\Projects\asfdgfhjghk\image caption\data\captions.pkl"
FEATURES_PATH = r"D:\Projects\asfdgfhjghk\image caption\data\image_features.pkl"

# Hyperparameters
EMBED_SIZE     = 256
HIDDEN_SIZE    = 256
ATTENTION_DIM  = 256
ENCODER_DIM    = 256
BATCH_SIZE     = 64
NUM_EPOCHS     = 15
LEARNING_RATE  = 8e-4
MAX_SEQ_LENGTH = 20
WEIGHT_DECAY   = 2e-4
PATIENCE       = 3
TEACHER_FORCING_RATIO = 0.5

# Reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print("Running train.py version: May 18, 2025, 19:42 IST (Improved Accuracy v2)")

    # 1) Load data
    try:
        with open(CAPTIONS_PATH, 'rb') as f:
            captions = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f:
            features = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        return

    # 2) Split IDs
    img_ids = list(captions.keys())
    random.shuffle(img_ids)
    n = len(img_ids)
    train_ids = img_ids[:int(0.8*n)]
    val_ids   = img_ids[int(0.8*n):int(0.9*n)]

    # 3) Dataset & Loader
    try:
        train_ds = Flickr8kDataset(train_ids, FEATURES_PATH, CAPTIONS_PATH, max_len=MAX_SEQ_LENGTH)
        val_ds   = Flickr8kDataset(val_ids, FEATURES_PATH, CAPTIONS_PATH, vocab=train_ds.vocab, max_len=MAX_SEQ_LENGTH)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # 4) Model
    encoder = EncoderCNN(EMBED_SIZE).to(device)
    for param in encoder.resnet.parameters():
        param.requires_grad = False
    for param in encoder.resnet[-2:].parameters():
        param.requires_grad = True
    decoder = DecoderWithAttention(EMBED_SIZE, HIDDEN_SIZE, len(train_ds.vocab), ATTENTION_DIM, ENCODER_DIM).to(device)
    pad_idx = train_ds.vocab.get('<pad>', 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Log vocab details
    print(f"Vocabulary size: {len(train_ds.vocab)}")
    print(f"Sample vocab keys: {list(train_ds.vocab.keys())[:10]}")
    print(f"Special tokens: <pad>={pad_idx}, <unk>={train_ds.vocab.get('<unk>')}, "
          f"<start>={train_ds.vocab.get('<start>')}, <end>={train_ds.vocab.get('<end>')}")
    print("Sample captions from captions.pkl:")
    for img, caps in list(captions.items())[:2]:
        for i, cap in enumerate(caps[:2]):
            print(f"{img} caption {i+1}: {cap}")

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    cider_scorer = Cider() if CIDER_AVAILABLE else None

    # 5) Training
    for epoch in range(1, NUM_EPOCHS+1):
        encoder.train()
        decoder.train()
        train_loss, train_acc = 0, 0
        for feats, caps, _ in train_loader:
            feats, caps = feats.to(device), caps.to(device)
            enc_out = encoder(feats)
            outputs, _ = decoder(enc_out, caps, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            targets = caps[:,1:]
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(2)
            mask = targets != pad_idx
            train_acc += ((preds==targets)&mask).sum().float()/mask.sum().float()
        avg_train_loss = train_loss/len(train_loader)
        avg_train_acc  = train_acc/len(train_loader)

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss, val_acc, bleu_scores, cider_scores = 0, 0, [], []
        with torch.no_grad():
            for batch_idx, (feats, caps, _) in enumerate(val_loader):
                feats, caps = feats.to(device), caps.to(device)
                enc_out = encoder(feats)
                outputs, alphas = decoder(enc_out, caps)
                targets = caps[:,1:]
                loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
                val_loss += loss.item()
                preds = outputs.argmax(2)
                mask = targets != pad_idx
                val_acc += ((preds==targets)&mask).sum().float()/mask.sum().float()
                # Compute BLEU and CIDEr
                gts, res = {}, {}
                for i, (pred, target) in enumerate(zip(preds, targets)):
                    pred_words, target_words = [], []
                    for idx in pred:
                        if idx == pad_idx or idx.item() not in train_ds.vocab:
                            continue
                        word = next((w for w, j in train_ds.vocab.items() if j == idx.item()), None)
                        if word:
                            pred_words.append(word)
                    for idx in target:
                        if idx == pad_idx or idx.item() not in train_ds.vocab:
                            continue
                        word = next((w for w, j in train_ds.vocab.items() if j == idx.item()), None)
                        if word:
                            target_words.append(word)
                    if pred_words and target_words:
                        bleu_scores.append(sentence_bleu([target_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25)))
                        if CIDER_AVAILABLE:
                            gts[i] = [target_words]
                            res[i] = [pred_words]
                if CIDER_AVAILABLE and gts and res:
                    try:
                        cider_score, _ = cider_scorer.compute_score(gts, res)
                        cider_scores.append(cider_score)
                    except Exception as e:
                        print(f"CIDEr computation error: {e}")
                # Save attention weights for first batch
                if batch_idx == 0:
                    np.save(f"attention_epoch_{epoch}_batch_{batch_idx}.npy", alphas.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = val_acc / len(val_loader)
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_cider = sum(cider_scores) / len(cider_scores) if cider_scores else 0
        print(f"Epoch {epoch}/{NUM_EPOCHS}  Train Loss={avg_train_loss:.4f}  "
              f"Train Acc={avg_train_acc:.4f}  Val Loss={avg_val_loss:.4f}  "
              f"Val Acc={avg_val_acc:.4f}  Val BLEU={avg_bleu:.4f}" +
              (f"  Val CIDEr={avg_cider:.4f}" if CIDER_AVAILABLE and cider_scores else ""))

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'vocab': train_ds.vocab
        }, f"checkpoint_{epoch}.pth")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        # Step scheduler
        scheduler.step(avg_val_loss)

if __name__=='__main__':
    train()