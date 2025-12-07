# 🎓 HƯỚNG DẪN CHUYỂN SANG GOOGLE COLAB

## 🎯 Tại sao nên dùng Google Colab?

Theo **Lưu ý quan trọng mục 11** của thầy:
> "Mã nguồn phải chạy được từ đầu đến cuối trên Google Colab hoặc máy local"

**Lợi ích:**
- ✅ GPU miễn phí (T4/P100) → Training nhanh hơn **10-20 lần**
- ✅ Không cần cài đặt môi trường phức tạp
- ✅ Export notebook (.ipynb) + PDF báo cáo dễ dàng
- ✅ Checkpoint lưu trực tiếp Google Drive

---

## 📋 BƯỚC 1: Tạo Notebook trên Colab

1. Truy cập: https://colab.research.google.com/
2. Chọn **File → New Notebook**
3. Đổi tên: `NLP_Do_An_EnFr_Translation.ipynb`
4. **BẬT GPU**: Runtime → Change runtime type → Hardware accelerator → **GPU** → Save

---

## 📁 BƯỚC 2: Upload Data lên Colab

### Cách 1: Upload trực tiếp (nhanh, nhưng mất khi runtime restart)

```python
# Cell 1: Upload data files
from google.colab import files
import os

# Tạo thư mục data
!mkdir -p /content/data

# Upload 6 files: train.en, train.fr, val.en, val.fr, test.en, test.fr
# Click "Choose Files" và chọn tất cả 6 files từ thư mục data/ trên máy
uploaded = files.upload()

# Di chuyển vào thư mục data
for filename in uploaded.keys():
    !mv {filename} /content/data/
    
print("✅ Đã upload xong data!")
```

### Cách 2: Lưu trên Google Drive (khuyến nghị - bền vững)

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Tạo thư mục project trên Drive
!mkdir -p "/content/drive/MyDrive/NLP_Do_An"
!mkdir -p "/content/drive/MyDrive/NLP_Do_An/data"
!mkdir -p "/content/drive/MyDrive/NLP_Do_An/check_point"

# Sau đó upload 6 files data vào:
# Google Drive → MyDrive → NLP_Do_An → data/
# (Kéo thả từ máy local vào Drive)

# Cell 3: Symbolic link
!ln -s "/content/drive/MyDrive/NLP_Do_An/data" /content/data
!ln -s "/content/drive/MyDrive/NLP_Do_An/check_point" /content/check_point

print("✅ Data sẵn sàng!")
```

---

## 🔧 BƯỚC 3: Cài đặt Dependencies

```python
# Cell 2: Install dependencies
!pip install -q spacy torch nltk matplotlib seaborn tqdm

# Download spaCy models
!python -m spacy download en_core_web_sm
!python -m spacy download fr_core_news_sm

print("✅ Cài đặt hoàn tất!")
```

---

## 📝 BƯỚC 4: Upload và Chạy File Config

### 🎯 Cách 1: Upload file config.py (KHUYẾN NGHỊ - Nhanh nhất)

```python
# Cell 3: Upload và chạy file config.py
from google.colab import files
import sys

# Upload file config.py từ máy local (trong thư mục src/)
print("📤 Chọn file config.py từ thư mục src/...")
uploaded = files.upload()

# Lưu vào thư mục hiện tại
!mkdir -p /content/src
for filename in uploaded.keys():
    with open(f'/content/src/{filename}', 'wb') as f:
        f.write(uploaded[filename])

# Import và chạy config
sys.path.append('/content/src')
from config import *

print(f"✅ Config đã load!")
print(f"🚀 Device: {DEVICE}")
print(f"📊 Batch size: {BATCH_SIZE}, Max vocab: {MAX_VOCAB_SIZE}")
```

### 🎯 Cách 2: Copy code từ config.py vào cell (Nếu không muốn upload file)

```python
# Cell 3: Configuration (copy từ src/config.py)
import torch
from pathlib import Path

# Paths
DATA_DIR = Path("/content/data")
CHECKPOINT_DIR = Path("/content/check_point")

# Vocabulary
MAX_VOCAB_SIZE = 10000
PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<sos>", "<eos>"
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
MIN_FREQ = 1

# Data files
TRAIN_EN = DATA_DIR / "train.en"
TRAIN_FR = DATA_DIR / "train.fr"
VAL_EN = DATA_DIR / "val.en"
VAL_FR = DATA_DIR / "val.fr"
TEST_EN = DATA_DIR / "test.en"
TEST_FR = DATA_DIR / "test.fr"

# Training config
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 50
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")
```
 
---

## 📝 BƯỚC 5: Upload và Chạy File Utils & Data Loader

### Cell 4: Upload utils.py

```python
# Cell 4: Upload và chạy utils.py
print("📤 Chọn file utils.py từ thư mục src/...")
uploaded = files.upload()

for filename in uploaded.keys():
    with open(f'/content/src/{filename}', 'wb') as f:
        f.write(uploaded[filename])

# Import utils
from utils import *
print("✅ Utils đã load!")
```

**HOẶC copy toàn bộ code từ `src/utils.py`** vào cell này (nếu không muốn upload).

### Cell 5: Upload data_loader.py

```python
# Cell 5: Upload và chạy data_loader.py
print("📤 Chọn file data_loader.py từ thư mục src/...")
uploaded = files.upload()

for filename in uploaded.keys():
    with open(f'/content/src/{filename}', 'wb') as f:
        f.write(uploaded[filename])

# Import data_loader
from data_loader import *
print("✅ Data loader đã load!")
```

**HOẶC copy toàn bộ code từ `src/data_loader.py`** vào cell này.

### Cell 6: Build Vocabularies

```python
# ============ BUILD VOCABULARIES ============
print("Building vocabularies...")
src_vocab, tgt_vocab = build_vocabularies(TRAIN_EN, TRAIN_FR, MAX_VOCAB_SIZE)

# Save vocabularies
save_vocab(src_vocab, CHECKPOINT_DIR / "src_vocab.pth")
save_vocab(tgt_vocab, CHECKPOINT_DIR / "tgt_vocab.pth")

print(f"✅ English vocab: {len(src_vocab)} tokens")
print(f"✅ French vocab: {len(tgt_vocab)} tokens")
```

### Cell 7: Prepare DataLoaders

```python
# ============ PREPARE DATALOADERS ============
train_loader, val_loader, test_loader = prepare_data_loaders(
    src_vocab, tgt_vocab, BATCH_SIZE
)

# Test một batch
for src_batch, src_lengths, tgt_batch, tgt_lengths in train_loader:
    print(f"✅ Source batch: {src_batch.shape}")
    print(f"✅ Target batch: {tgt_batch.shape}")
    print(f"✅ Source lengths (sorted): {src_lengths[:5]}")
    break
```

---

## 🏗️ BƯỚC 5: Model (Task 3 - Chưa có code)

### Cell 8: Encoder

```python
# ============ ENCODER ============
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_lengths):
        # src: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(src))
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Context vector = hidden state của layer cuối
        return hidden, cell
```

### Cell 9: Decoder

```python
# ============ DECODER ============
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        # input: (batch_size, 1)
        input = input.unsqueeze(1) if input.dim() == 1 else input
        
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell
```

### Cell 10: Seq2Seq

```python
# ============ SEQ2SEQ ============
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encoder
        hidden, cell = self.encoder(src, src_lengths)
        
        # Decoder input: <sos>
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
        
        return outputs

# Initialize model
encoder = Encoder(len(src_vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(tgt_vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

---

## 🏋️ BƯỚC 6: Training (Task 4)

```python
# ============ TRAINING ============
import torch.optim as optim
from tqdm import tqdm

# Loss & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for src, src_len, tgt, tgt_len in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        output = model(src, src_len, tgt, TEACHER_FORCING_RATIO)
        
        # Reshape for loss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# Validation function
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, src_len, tgt, tgt_len in loader:
            output = model(src, src_len, tgt, 0)  # No teacher forcing
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# Training loop
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
        print("✅ Saved best model!")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("⚠️ Early stopping!")
            break
    
    scheduler.step(val_loss)

print("✅ Training completed!")
```

---

## 📊 BƯỚC 7: Visualization

```python
# ============ PLOT LOSSES ============
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')
plt.grid(True)
plt.show()
```

---

## 🔍 BƯỚC 8: Evaluation & Translation

```python
# ============ TRANSLATE FUNCTION ============
def translate(sentence, src_vocab, tgt_vocab, model, device, max_len=50):
    model.eval()
    
    # Tokenize
    tokens = tokenize_sentence(sentence)
    tokens = add_special_tokens(tokens, add_sos=True, add_eos=True)
    
    # Encode
    src_indices = src_vocab.encode(tokens)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(src_indices)])
    
    with torch.no_grad():
        # Encoder
        hidden, cell = model.encoder(src_tensor, src_len)
        
        # Decoder (greedy)
        input = torch.LongTensor([tgt_vocab.sos_idx]).to(device)
        output_tokens = []
        
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input, hidden, cell)
            pred_token = output.argmax(1).item()
            
            if pred_token == tgt_vocab.eos_idx:
                break
            
            output_tokens.append(pred_token)
            input = torch.LongTensor([pred_token]).to(device)
    
    # Decode
    translated = tgt_vocab.decode(output_tokens)
    return ' '.join(translated)

# Test
test_sentences = [
    "A man is walking in the street.",
    "The cat is sleeping on the bed.",
    "I love programming."
]

for sent in test_sentences:
    translated = translate(sent, src_vocab, tgt_vocab, model, DEVICE)
    print(f"EN: {sent}")
    print(f"FR: {translated}")
    print()
```

---

## 📥 BƯỚC 9: Export Notebook

1. **Download notebook**: File → Download → Download .ipynb
2. **Export PDF**: File → Print → Save as PDF
3. **Download checkpoint**: 
   ```python
   from google.colab import files
   files.download('/content/check_point/best_model.pth')
   files.download('/content/check_point/src_vocab.pth')
   files.download('/content/check_point/tgt_vocab.pth')
   ```

---

## ✅ CHECKLIST HOÀN TẤT

- [ ] Tạo notebook trên Colab
- [ ] Bật GPU (Runtime → Change runtime type)
- [ ] Upload data lên Colab
- [ ] Cài đặt dependencies
- [ ] Copy code Task 1-2 (config, utils, data_loader)
- [ ] Implement Task 3 (Encoder-Decoder model)
- [ ] Implement Task 4 (Training loop)
- [ ] Implement Task 5 (translate function)
- [ ] Test translate trên vài câu
- [ ] Đánh giá BLEU score trên test set
- [ ] Vẽ biểu đồ train/val loss
- [ ] Phân tích 5 ví dụ lỗi dịch
- [ ] Download notebook (.ipynb)
- [ ] Export PDF báo cáo
- [ ] Download checkpoint files

---

## 🎯 LƯU Ý QUAN TRỌNG

1. **GPU Runtime**: Colab free có giới hạn ~12 giờ/session. Nên:
   - Save checkpoint thường xuyên
   - Mount Google Drive để lưu checkpoint tự động

2. **Disconnect**: Nếu bị disconnect, chạy lại từ Cell "Mount Drive" là có thể load lại checkpoint.

3. **Báo cáo PDF**: 
   - Phải có: sơ đồ kiến trúc, biểu đồ loss, BLEU score, 5 ví dụ dịch + phân tích
   - In trực tiếp từ notebook hoặc viết riêng trong Word/LaTeX

4. **Nộp bài**: 
   - 01 file PDF báo cáo (đầy đủ nội dung)
   - Mã nguồn: notebook (.ipynb) + checkpoint (.pth) (nén thành .zip)

---

**Good luck! 🚀**
