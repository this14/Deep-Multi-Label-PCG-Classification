"""
Enhanced Multi-branch CNN Training with Data Augmentation
Dataset: CirCor DigiScope Phonocardiogram Dataset (Small Dataset Optimized)
Task: Multi-label classification of Timing, Shape, and Grading
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = r"D:\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
    CSV_PATH = r"D:\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
    OUTPUT_DIR = r"D:\the-circor-digiscope-phonocardiogram-dataset-1.0.3\model_outputs"
    
    SAMPLE_RATE = 4000
    DURATION = 5
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.5
    
    LOCATIONS = ['AV', 'PV', 'TV', 'MV']

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Data Augmentation
# ============================================================================
class AudioAugmentor:
    @staticmethod
    def add_gaussian_noise(audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    @staticmethod
    def time_shift(audio, shift_max=0.2):
        shift = np.random.randint(int(len(audio) * shift_max))
        direction = np.random.choice([-1, 1])
        return np.roll(audio, shift * direction)
    
    @staticmethod
    def pitch_shift(audio, sr, n_steps=2):
        n_steps = np.random.uniform(-n_steps, n_steps)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def apply_random_augmentation(audio, sr):
        augmentations = [
            lambda x: AudioAugmentor.add_gaussian_noise(x, 0.005),
            lambda x: AudioAugmentor.time_shift(x, 0.2),
            lambda x: AudioAugmentor.pitch_shift(x, sr, 1),
        ]
        
        n_augs = np.random.randint(1, 3)
        selected = np.random.choice(len(augmentations), n_augs, replace=False)
        
        for idx in selected:
            audio = augmentations[idx](audio)
        
        return audio

# ============================================================================
# Dataset
# ============================================================================
class HeartSoundDataset(Dataset):
    def __init__(self, data_df, data_dir, augment=False):
        self.data_df = data_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.augment = augment
        self.augmentor = AudioAugmentor()
        self.valid_samples = self._prepare_samples()
        
    def _prepare_samples(self):
        valid_samples = []
        for idx, row in self.data_df.iterrows():
            if row['Murmur'] != 'Present':
                continue
            if (pd.notna(row['Systolic murmur timing']) and 
                pd.notna(row['Systolic murmur shape']) and 
                pd.notna(row['Systolic murmur grading'])):
                
                valid_samples.append({
                    'patient_id': row['Patient ID'],
                    'locations': row['Recording locations:'].split('+'),
                    'timing': row['Systolic murmur timing'],
                    'shape': row['Systolic murmur shape'],
                    'grading': row['Systolic murmur grading']
                })
        return valid_samples
    
    def _load_and_preprocess_audio(self, file_path, apply_augmentation=False):
        try:
            if not os.path.exists(file_path):
                return None
            
            audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION)
            
            if apply_augmentation and Config.USE_AUGMENTATION:
                if np.random.random() < Config.AUGMENTATION_PROB:
                    audio = self.augmentor.apply_random_augmentation(audio, sr)
            
            target_length = Config.SAMPLE_RATE * Config.DURATION
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            audio = librosa.effects.preemphasis(audio)
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=Config.SAMPLE_RATE, n_mels=Config.N_MELS,
                n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
            
            return mel_spec_db
        except:
            return None
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        patient_id = sample['patient_id']
        locations = sample['locations']
        
        spectrograms = []
        for loc in Config.LOCATIONS:
            spec = None
            if loc in locations:
                file_path = os.path.join(self.data_dir, f"{patient_id}_{loc}.wav")
                spec = self._load_and_preprocess_audio(file_path, self.augment)
            
            if spec is None:
                for available_loc in locations:
                    file_path = os.path.join(self.data_dir, f"{patient_id}_{available_loc}.wav")
                    spec = self._load_and_preprocess_audio(file_path, self.augment)
                    if spec is not None:
                        break
            
            if spec is None:
                spec = np.zeros((Config.N_MELS, 40))
            
            spectrograms.append(spec)
        
        spectrograms = torch.FloatTensor(np.stack(spectrograms, axis=0))
        
        return {
            'spectrogram': spectrograms,
            'timing': sample['timing'],
            'shape': sample['shape'],
            'grading': sample['grading'],
            'patient_id': patient_id
        }

# ============================================================================
# Model
# ============================================================================
class MultiBranchCNN(nn.Module):
    def __init__(self, num_timing_classes, num_shape_classes, num_grading_classes):
        super(MultiBranchCNN, self).__init__()
        
        self.branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.5)
        )
        
        self.feature_dim = 128 * 4 * 4
        
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.timing_head = nn.Linear(256, num_timing_classes)
        self.shape_head = nn.Linear(256, num_shape_classes)
        self.grading_head = nn.Linear(256, num_grading_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        branch_features = []
        
        for i in range(4):
            branch_input = x[:, i:i+1, :, :]
            branch_output = self.branch(branch_input)
            branch_output = branch_output.view(batch_size, -1)
            branch_features.append(branch_output)
        
        fused_features = torch.cat(branch_features, dim=1)
        fused = self.fusion(fused_features)
        
        return self.timing_head(fused), self.shape_head(fused), self.grading_head(fused)

# ============================================================================
# Training
# ============================================================================
def train_epoch(model, loader, crits, optimizer, device, encoders):
    model.train()
    running_loss = 0.0
    all_preds = {'timing': [], 'shape': [], 'grading': []}
    all_labels = {'timing': [], 'shape': [], 'grading': []}
    
    for batch in tqdm(loader, desc="Training"):
        spec = batch['spectrogram'].to(device)
        
        t_labels = torch.LongTensor([encoders[0].transform([t])[0] for t in batch['timing']]).to(device)
        s_labels = torch.LongTensor([encoders[1].transform([s])[0] for s in batch['shape']]).to(device)
        g_labels = torch.LongTensor([encoders[2].transform([g])[0] for g in batch['grading']]).to(device)
        
        optimizer.zero_grad()
        t_out, s_out, g_out = model(spec)
        
        loss = crits[0](t_out, t_labels) + crits[1](s_out, s_labels) + crits[2](g_out, g_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        all_preds['timing'].extend(torch.argmax(t_out, dim=1).cpu().numpy())
        all_labels['timing'].extend(t_labels.cpu().numpy())
        all_preds['shape'].extend(torch.argmax(s_out, dim=1).cpu().numpy())
        all_labels['shape'].extend(s_labels.cpu().numpy())
        all_preds['grading'].extend(torch.argmax(g_out, dim=1).cpu().numpy())
        all_labels['grading'].extend(g_labels.cpu().numpy())
    
    return (running_loss / len(loader),
            f1_score(all_labels['timing'], all_preds['timing'], average='macro', zero_division=0),
            f1_score(all_labels['shape'], all_preds['shape'], average='macro', zero_division=0),
            f1_score(all_labels['grading'], all_preds['grading'], average='macro', zero_division=0))

def validate(model, loader, crits, device, encoders):
    model.eval()
    running_loss = 0.0
    all_preds = {'timing': [], 'shape': [], 'grading': []}
    all_labels = {'timing': [], 'shape': [], 'grading': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            spec = batch['spectrogram'].to(device)
            
            t_labels = torch.LongTensor([encoders[0].transform([t])[0] for t in batch['timing']]).to(device)
            s_labels = torch.LongTensor([encoders[1].transform([s])[0] for s in batch['shape']]).to(device)
            g_labels = torch.LongTensor([encoders[2].transform([g])[0] for g in batch['grading']]).to(device)
            
            t_out, s_out, g_out = model(spec)
            
            loss = crits[0](t_out, t_labels) + crits[1](s_out, s_labels) + crits[2](g_out, g_labels)
            running_loss += loss.item()
            
            all_preds['timing'].extend(torch.argmax(t_out, dim=1).cpu().numpy())
            all_labels['timing'].extend(t_labels.cpu().numpy())
            all_preds['shape'].extend(torch.argmax(s_out, dim=1).cpu().numpy())
            all_labels['shape'].extend(s_labels.cpu().numpy())
            all_preds['grading'].extend(torch.argmax(g_out, dim=1).cpu().numpy())
            all_labels['grading'].extend(g_labels.cpu().numpy())
    
    return {
        'loss': running_loss / len(loader),
        'timing_f1': f1_score(all_labels['timing'], all_preds['timing'], average='macro', zero_division=0),
        'timing_acc': accuracy_score(all_labels['timing'], all_preds['timing']),
        'shape_f1': f1_score(all_labels['shape'], all_preds['shape'], average='macro', zero_division=0),
        'shape_acc': accuracy_score(all_labels['shape'], all_preds['shape']),
        'grading_f1': f1_score(all_labels['grading'], all_preds['grading'], average='macro', zero_division=0),
        'grading_acc': accuracy_score(all_labels['grading'], all_preds['grading']),
        'timing_preds': all_preds['timing'], 'timing_labels': all_labels['timing'],
        'shape_preds': all_preds['shape'], 'shape_labels': all_labels['shape'],
        'grading_preds': all_preds['grading'], 'grading_labels': all_labels['grading']
    }

# ============================================================================
# Visualization
# ============================================================================
def plot_confusion_matrices(results, encoders, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cm_t = confusion_matrix(results['timing_labels'], results['timing_preds'])
    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=encoders[0].classes_, yticklabels=encoders[0].classes_)
    axes[0].set_title('Timing', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    cm_s = confusion_matrix(results['shape_labels'], results['shape_preds'])
    sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=encoders[1].classes_, yticklabels=encoders[1].classes_)
    axes[1].set_title('Shape', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    cm_g = confusion_matrix(results['grading_labels'], results['grading_preds'])
    sns.heatmap(cm_g, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=encoders[2].classes_, yticklabels=encoders[2].classes_)
    axes[2].set_title('Grading', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved")

def plot_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0,0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0,0].set_title('Loss', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    axes[0,1].plot(history['train_timing_f1'], label='Train', linewidth=2)
    axes[0,1].plot(history['val_timing_f1'], label='Val', linewidth=2)
    axes[0,1].set_title('Timing F1', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    axes[1,0].plot(history['train_shape_f1'], label='Train', linewidth=2)
    axes[1,0].plot(history['val_shape_f1'], label='Val', linewidth=2)
    axes[1,0].set_title('Shape F1', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    axes[1,1].plot(history['train_grading_f1'], label='Train', linewidth=2)
    axes[1,1].plot(history['val_grading_f1'], label='Val', linewidth=2)
    axes[1,1].set_title('Grading F1', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved")

# ============================================================================
# Main
# ============================================================================
def main():
    print("="*80)
    print("Enhanced Training (Small Dataset Optimized)")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Augmentation: {'ON' if Config.USE_AUGMENTATION else 'OFF'}\n")
    
    df = pd.read_csv(Config.CSV_PATH)
    full_dataset = HeartSoundDataset(df, Config.DATA_DIR, augment=False)
    print(f"Valid samples: {len(full_dataset)}\n")
    
    if len(full_dataset) < 20:
        print("ERROR: Too few samples!")
        return
    
    all_timing = [s['timing'] for s in full_dataset.valid_samples]
    all_shape = [s['shape'] for s in full_dataset.valid_samples]
    all_grading = [s['grading'] for s in full_dataset.valid_samples]
    
    enc_t = LabelEncoder().fit(all_timing)
    enc_s = LabelEncoder().fit(all_shape)
    enc_g = LabelEncoder().fit(all_grading)
    encoders = [enc_t, enc_s, enc_g]
    
    print(f"Classes:")
    print(f"  Timing ({len(enc_t.classes_)}): {list(enc_t.classes_)}")
    print(f"  Shape ({len(enc_s.classes_)}): {list(enc_s.classes_)}")
    print(f"  Grading ({len(enc_g.classes_)}): {list(enc_g.classes_)}\n")
    
    # Check class distribution
    timing_enc = enc_t.transform(all_timing)
    unique, counts = np.unique(timing_enc, return_counts=True)
    print("Class distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {enc_t.classes_[cls_idx]}: {count} samples")
    print()
    
    # Remove rare classes (< 2 samples) for stratification
    rare_classes = unique[counts < 2]
    if len(rare_classes) > 0:
        print(f"⚠️ Warning: Removing {len(rare_classes)} rare classes for stratification")
        rare_class_names = [enc_t.classes_[idx] for idx in rare_classes]
        print(f"  Rare classes: {rare_class_names}\n")
        
        # Filter out rare classes
        mask = np.isin(timing_enc, rare_classes, invert=True)
        indices = np.arange(len(full_dataset))[mask]
        timing_enc_filtered = timing_enc[mask]
    else:
        indices = np.arange(len(full_dataset))
        timing_enc_filtered = timing_enc
    
    # Use simple random split if still problematic, otherwise stratified
    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(sss1.split(indices, timing_enc_filtered))
        train_idx = indices[train_idx]
        temp_idx = indices[temp_idx]
        
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx_rel, test_idx_rel = next(sss2.split(np.arange(len(temp_idx)), timing_enc_filtered[np.isin(indices, temp_idx)]))
        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]
        print("✓ Using stratified split")
    except:
        # Fallback to simple random split
        print("⚠️ Using simple random split (stratification failed)")
        np.random.seed(42)
        indices = np.arange(len(full_dataset))
        np.random.shuffle(indices)
        
        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
    
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}\n")
    
    train_ds = HeartSoundDataset(df.iloc[train_idx], Config.DATA_DIR, augment=True)
    val_ds = HeartSoundDataset(df.iloc[val_idx], Config.DATA_DIR, augment=False)
    test_ds = HeartSoundDataset(df.iloc[test_idx], Config.DATA_DIR, augment=False)
    
    train_loader = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = MultiBranchCNN(len(enc_t.classes_), len(enc_s.classes_), len(enc_g.classes_)).to(Config.DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")
    
    w_t = compute_class_weight('balanced', classes=np.unique(timing_enc), y=timing_enc)
    w_s = compute_class_weight('balanced', classes=np.unique(enc_s.transform(all_shape)), y=enc_s.transform(all_shape))
    w_g = compute_class_weight('balanced', classes=np.unique(enc_g.transform(all_grading)), y=enc_g.transform(all_grading))
    
    crit_t = nn.CrossEntropyLoss(weight=torch.FloatTensor(w_t).to(Config.DEVICE))
    crit_s = nn.CrossEntropyLoss(weight=torch.FloatTensor(w_s).to(Config.DEVICE))
    crit_g = nn.CrossEntropyLoss(weight=torch.FloatTensor(w_g).to(Config.DEVICE))
    crits = [crit_t, crit_s, crit_g]
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_timing_f1': [], 'val_timing_f1': [],
        'train_shape_f1': [], 'val_shape_f1': [],
        'train_grading_f1': [], 'val_grading_f1': []
    }
    
    best_f1 = 0.0
    patience = 0
    
    print("Training started...\n")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-"*80)
        
        train_loss, train_t_f1, train_s_f1, train_g_f1 = train_epoch(
            model, train_loader, crits, optimizer, Config.DEVICE, encoders)
        
        val_res = validate(model, val_loader, crits, Config.DEVICE, encoders)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_res['loss'])
        history['train_timing_f1'].append(train_t_f1)
        history['val_timing_f1'].append(val_res['timing_f1'])
        history['train_shape_f1'].append(train_s_f1)
        history['val_shape_f1'].append(val_res['shape_f1'])
        history['train_grading_f1'].append(train_g_f1)
        history['val_grading_f1'].append(val_res['grading_f1'])
        
        print(f"Loss: Train={train_loss:.4f}, Val={val_res['loss']:.4f}")
        print(f"Timing  F1: {train_t_f1:.4f}/{val_res['timing_f1']:.4f} (Acc: {val_res['timing_acc']:.4f})")
        print(f"Shape   F1: {train_s_f1:.4f}/{val_res['shape_f1']:.4f} (Acc: {val_res['shape_acc']:.4f})")
        print(f"Grading F1: {train_g_f1:.4f}/{val_res['grading_f1']:.4f} (Acc: {val_res['grading_acc']:.4f})")
        
        avg_f1 = (val_res['timing_f1'] + val_res['shape_f1'] + val_res['grading_f1']) / 3
        scheduler.step(avg_f1)
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience = 0
            torch.save({'model': model.state_dict(), 'f1': avg_f1}, 
                      os.path.join(Config.OUTPUT_DIR, 'best_model.pth'))
            print(f"✓ Best (Avg F1: {avg_f1:.4f})")
        else:
            patience += 1
            if patience >= 20:
                print(f"\nEarly stop at epoch {epoch+1}")
                break
        print()
    
    print("="*80)
    print("Test Evaluation")
    print("="*80)
    
    ckpt = torch.load(os.path.join(Config.OUTPUT_DIR, 'best_model.pth'), 
                      map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    test_res = validate(model, test_loader, crits, Config.DEVICE, encoders)
    
    print("\nTest Results:")
    print("-"*80)
    print(f"Timing  - F1: {test_res['timing_f1']:.4f}, Acc: {test_res['timing_acc']:.4f}")
    print(f"Shape   - F1: {test_res['shape_f1']:.4f}, Acc: {test_res['shape_acc']:.4f}")
    print(f"Grading - F1: {test_res['grading_f1']:.4f}, Acc: {test_res['grading_acc']:.4f}")
    print(f"\nAverage F1: {(test_res['timing_f1']+test_res['shape_f1']+test_res['grading_f1'])/3:.4f}")
    
    print("\n" + "="*80)
    print("Classification Reports")
    print("="*80)
    
    print("\nTiming:")
    print(classification_report(test_res['timing_labels'], test_res['timing_preds'],
                                labels=range(len(enc_t.classes_)),
                                target_names=enc_t.classes_, zero_division=0))
    
    print("\nShape:")
    print(classification_report(test_res['shape_labels'], test_res['shape_preds'],
                                labels=range(len(enc_s.classes_)),
                                target_names=enc_s.classes_, zero_division=0))
    
    print("\nGrading:")
    print(classification_report(test_res['grading_labels'], test_res['grading_preds'],
                                labels=range(len(enc_g.classes_)),
                                target_names=enc_g.classes_, zero_division=0))
    
    results_df = pd.DataFrame({
        'Task': ['Timing', 'Shape', 'Grading'],
        'Macro-F1': [test_res['timing_f1'], test_res['shape_f1'], test_res['grading_f1']],
        'Accuracy': [test_res['timing_acc'], test_res['shape_acc'], test_res['grading_acc']]
    })
    results_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'test_results.csv'), index=False)
    
    plot_confusion_matrices(test_res, encoders, os.path.join(Config.OUTPUT_DIR, 'confusion_matrices.png'))
    plot_history(history, os.path.join(Config.OUTPUT_DIR, 'training_history.png'))
    
    print("\n" + "="*80)
    print(f"Complete! Results in: {Config.OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()