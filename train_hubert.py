import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import (Wav2Vec2FeatureExtractor,
                          HubertForSequenceClassification,
                          AdamW)
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------
# 1. Load Your DataFrame
# ------------------------------------------------------------------
df = pd.read_csv("iemocap_dataframe.csv")

# Filter out any noise types or SNR levels not wanted, if necessary:
VALID_NOISE_TYPES = ["babble", "music", "noise", "white"]
VALID_SNR = ["-5", "0", "5", "10", "15", "20"]
VALID_SUBSETS = ["impro", "script", "full"]

df = df[df["noise"].isin(VALID_NOISE_TYPES) & df["snr"].isin(VALID_SNR)]

# ------------------------------------------------------------------
# 2. Load Audio & Tokenize
# ------------------------------------------------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")

def load_and_tokenize(wav_paths, sr=16000):
    """
    Loads each audio file in wav_paths, resamples to sr=16000, 
    then uses Wav2Vec2FeatureExtractor (or HuBERT feature_extractor) 
    to produce input_values, attention_mask.
    Returns a dict suitable for creating a torch Dataset.
    """
    all_waveforms = []
    for wav_path in wav_paths:
        speech, orig_sr = librosa.load(wav_path, sr=sr, mono=True)
        all_waveforms.append(speech)

    inputs = feature_extractor(
        all_waveforms,
        sampling_rate=sr,
        padding=True,
        return_tensors="pt"
    )
    return inputs

# ------------------------------------------------------------------
# 3. Define PyTorch Dataset for Emotion Recognition
# ------------------------------------------------------------------
class EmotionDataset(Dataset):
    """
    A PyTorch Dataset that holds extracted input_values, attention_mask, and labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# ------------------------------------------------------------------
# 4. Train Function (Handles a single noise/SNR/subset combination)
# ------------------------------------------------------------------
def train_hubert_for_noise_snr_subset(df, noise, snr, subset, device="cuda"):
    """
    Filters df for the given (noise, snr, subset), does a stratified split
    into train/val/test, fine-tunes a HuBERT model, and returns WA & UA metrics.
    """
    sub_df = df[(df["noise"] == noise) & (df["snr"] == snr) & (df["subset"] == subset)]
    if len(sub_df) < 2:
        print(f"[Warning] No data for {noise}/{snr}/{subset}; skipping.")
        return None

    print(f"=== Training HuBERT for {noise}/{snr}/{subset}, total samples: {len(sub_df)} ===")

    # Get file paths & labels
    wavfiles = sub_df["utterance_path"].tolist()
    labels   = sub_df["label"].tolist()
    
    # Stratified Split (80% train, 10% val, 10% test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        wavfiles, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    print(f"Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Tokenize/Encode Audio
    train_inputs = load_and_tokenize(train_files)
    val_inputs   = load_and_tokenize(val_files)
    test_inputs  = load_and_tokenize(test_files)

    # Wrap in PyTorch Dataset
    train_dataset = EmotionDataset(train_inputs, train_labels)
    val_dataset   = EmotionDataset(val_inputs, val_labels)
    test_dataset  = EmotionDataset(test_inputs, test_labels)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize Model
    model = HubertForSequenceClassification.from_pretrained(
        "superb/hubert-large-superb-er",
        num_labels=4  # We have 4 emotion labels: neutral(0), sad(1), angry(2), happy(3)
    )
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training Loop
    EPOCHS = 3
    model.train()
    for epoch_i in range(EPOCHS):
        print(f"Epoch {epoch_i+1}/{EPOCHS}")
        running_correct = 0
        running_total = 0
        epoch_loss_list = []

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(input_values, attention_mask=attention_mask, labels=labels_tensor)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            # Compute training accuracy
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels_tensor).sum().item()
            running_total   += labels_tensor.size(0)
            epoch_loss_list.append(loss.item())

        train_acc = running_correct / running_total
        avg_loss = np.mean(epoch_loss_list)
        print(f"Train Epoch={epoch_i+1}, Loss={avg_loss:.3f}, Accuracy={train_acc:.3f}")

        # Evaluate on Validation Set
        val_wa, val_ua = evaluate_model(model, val_loader, device)
        print(f"Val WA={val_wa:.3f}, Val UA={val_ua:.3f}")

    # Final Test Evaluation
    test_wa, test_ua = evaluate_model(model, test_loader, device)
    print(f"=== Final Test Results for {noise}/{snr}/{subset} ===")
    print(f"Test WA={test_wa:.3f}, Test UA={test_ua:.3f}")

    return {
        "noise": noise,
        "snr": snr,
        "subset": subset,
        "train_size": len(train_files),
        "val_size": len(val_files),
        "test_size": len(test_files),
        "final_WA": test_wa,
        "final_UA": test_ua
    }

# ------------------------------------------------------------------
# 5. Define Evaluation Function (WA and UA)
# ------------------------------------------------------------------
def evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=1)

            correct += (preds == labels_tensor).sum().item()
            total += labels_tensor.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())

    WA = correct / total if total > 0 else 0.0
    UA = np.mean([accuracy_score(np.array(all_labels) == i, np.array(all_preds) == i) for i in np.unique(all_labels)])
    
    return WA, UA

# ------------------------------------------------------------------
# 6. Train for Each (noise, snr, subset)
# ------------------------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
all_results = []

for noise in VALID_NOISE_TYPES:
    for snr in VALID_SNR:
        for sbst in VALID_SUBSETS:
            result = train_hubert_for_noise_snr_subset(df, noise, snr, sbst, device=DEVICE)
            if result:
                all_results.append(result)

pd.DataFrame(all_results).to_csv("hubert_experiment_results.csv", index=False)
print("Experiments done! WA and UA saved to hubert_experiment_results.csv")
