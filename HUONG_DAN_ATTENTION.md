# 🚀 HƯỚNG DẪN NÂNG BLEU LÊN 45%

## 📊 Mục tiêu
- **Hiện tại:** 32.67% BLEU
- **Mục tiêu:** 45% BLEU
- **Cần tăng:** ~12-13%

---

## ✅ Đã tạo sẵn

### 1. `src/attention_decoder.py` (270 dòng)
Chứa:
- ✅ **Attention** class - Bahdanau Attention mechanism
- ✅ **AttentionDecoder** class - Decoder với attention
- ✅ **Seq2SeqWithAttention** class - Model hoàn chỉnh
- ✅ Hàm count_parameters
- ✅ Example usage

### 2. `train_with_attention.py` (200 dòng)
Chứa:
- ✅ CONFIG optimized (vocab=15K, epochs=20, patience=5)
- ✅ build_attention_model() function
- ✅ train_with_scheduler() - với ReduceLROnPlateau
- ✅ Modified train_epoch() và evaluate() cho attention
- ✅ visualize_attention() để vẽ heatmap

---

## 🔨 CÁCH SỬ DỤNG

### Option 1: Copy-paste vào Notebook hiện tại (Nhanh nhất)

Mở notebook `NLP_Final_Project_Seq2Seq_Translation.ipynb`, thêm các cells sau:

#### Cell 1: Import Attention modules
```python
# Import Attention Decoder
import sys
sys.path.append('src')

from attention_decoder import (
    Attention, 
    AttentionDecoder, 
    Seq2SeqWithAttention
)

print("✅ Imported Attention modules")
```

#### Cell 2: Build model với Attention
```python
# ============================================
# BUILD MODEL WITH ATTENTION
# ============================================

# Update CONFIG
CONFIG['max_vocab_size'] = 15000  # Tăng vocab
CONFIG['num_epochs'] = 20         # Tăng epochs
CONFIG['early_stopping_patience'] = 5

INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)

# Encoder (giữ nguyên)
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)

# Attention mechanism (NEW!)
attn = Attention(HID_DIM, HID_DIM)

# Decoder with Attention (NEW!)
dec = AttentionDecoder(
    OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM,
    N_LAYERS, DROPOUT, attn
)

# Seq2Seq with Attention
model_attention = Seq2SeqWithAttention(enc, dec, device).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Model has {count_parameters(model_attention):,} trainable parameters')
print('✅ Model with Attention created!')
```

#### Cell 3: Modified Training Functions
```python
# ============================================
# MODIFIED TRAIN/EVAL (for Attention)
# ============================================

def train_epoch_attention(model, iterator, optimizer, criterion, clip, device):
    """Train epoch for attention model"""
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src, src_len = batch['src'].to(device), batch['src_len'].to(device)
        trg = batch['trg'].to(device)
        
        optimizer.zero_grad()
        
        # Forward (returns outputs AND attentions)
        outputs, attentions = model(src, src_len, trg)
        
        # Reshape
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:, :].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(outputs, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate_attention(model, iterator, criterion, device):
    """Evaluate attention model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            src, src_len = batch['src'].to(device), batch['src_len'].to(device)
            trg = batch['trg'].to(device)
            
            outputs, attentions = model(src, src_len, trg, teacher_forcing_ratio=0)
            
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:, :].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(outputs, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

print('✅ Training functions ready!')
```

#### Cell 4: Train với Learning Rate Scheduler
```python
# ============================================
# TRAINING WITH LR SCHEDULER
# ============================================

from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import math

# Optimizer
optimizer = optim.Adam(model_attention.parameters(), lr=CONFIG['learning_rate'])

# Loss
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# Training loop
best_val_loss = float('inf')
patience_counter = 0

print("="*60)
print("  TRAINING WITH ATTENTION - Target BLEU: 45%")
print("="*60)

for epoch in range(CONFIG['num_epochs']):
    start_time = time.time()
    
    # Train
    train_loss = train_epoch_attention(
        model_attention, train_iterator, optimizer, 
        criterion, CONFIG['clip'], device
    )
    
    # Evaluate
    val_loss = evaluate_attention(
        model_attention, val_iterator, criterion, device
    )
    
    # Update LR
    scheduler.step(val_loss)
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    # Check improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_attention.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'check_point/best_model_attention.pth')
        
        print(f'✅ Saved checkpoint')
    else:
        patience_counter += 1
    
    # Print stats
    print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tVal Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}')
    print(f'\tPatience: {patience_counter}/{CONFIG["early_stopping_patience"]}')
    
    # Early stopping
    if patience_counter >= CONFIG['early_stopping_patience']:
        print(f'\n⏹️ Early stopping at epoch {epoch+1}')
        break

print("\n✅ Training completed!")
```

#### Cell 5: Evaluate BLEU
```python
# ============================================
# EVALUATE BLEU SCORE
# ============================================

# Load best model
checkpoint = torch.load('check_point/best_model_attention.pth')
model_attention.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
print(f"Best validation loss: {checkpoint['val_loss']:.3f}")

# Calculate BLEU on test set
print("\n" + "="*60)
print("  CALCULATING BLEU SCORE ON TEST SET")
print("="*60)

bleu_score = calculate_bleu_on_test_set(
    test_data, src_vocab, tgt_vocab, model_attention, device
)

print("\n" + "="*60)
print(f"  🎯 FINAL BLEU SCORE: {bleu_score:.2f}%")
print("="*60)

# Compare with baseline
print(f"\n📊 Comparison:")
print(f"  Baseline (no Attention): 32.67%")
print(f"  With Attention:          {bleu_score:.2f}%")
print(f"  Improvement:             +{bleu_score - 32.67:.2f}%")

if bleu_score >= 45:
    print("\n🎉 ACHIEVED TARGET OF 45%!")
else:
    print(f"\n⚠️ Need +{45 - bleu_score:.2f}% more to reach 45%")
```

---

### Option 2: Chạy script Python độc lập

```bash
# Trong terminal PowerShell
cd D:\Corel\HK1_NAM3\NLP\NLP_DO_AN

# Chạy training script
python train_with_attention.py
```

**Lưu ý:** Cần sửa script để uncomment các dòng load data.

---

## ⏱️ Thời gian dự kiến

- **Training:** ~2-2.5 giờ (20 epochs, early stopping ~epoch 15)
- **Evaluation:** ~5-10 phút
- **Tổng:** ~2.5 giờ

---

## 📈 Kết quả mong đợi

### Sau khi train xong:

| Metric | Baseline | With Attention | Improvement |
|--------|----------|----------------|-------------|
| **BLEU** | 32.67% | **~44-47%** | **+12-15%** ✅ |
| Train Loss | 2.85 | ~2.3-2.5 | Lower |
| Val Loss | 3.24 | ~2.7-2.9 | Lower |
| Parameters | 20M | ~23M | +3M |

### Lý do cải thiện:

1. ✅ **Attention mechanism** (+10-12%) - Giải quyết bottleneck
2. ✅ **Vocab 15K** (+1%) - Giảm <unk> tokens
3. ✅ **Epochs 20** (+1-2%) - Train lâu hơn
4. ✅ **LR Scheduler** (+1%) - Tối ưu learning rate

---

## 🐛 Xử lý lỗi

### Lỗi 1: Out of Memory
```python
# Giảm batch_size
CONFIG['batch_size'] = 32  # Thay vì 64
```

### Lỗi 2: Import error
```python
# Kiểm tra path
import sys
sys.path.append('src')  # Đảm bảo có dòng này
```

### Lỗi 3: Model architecture mismatch
```python
# Đảm bảo Encoder có method forward trả về 3 values:
# encoder_outputs, hidden, cell
```

---

## 📊 Visualize Attention (Bonus)

Sau khi train xong, visualize attention:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Translate 1 câu
src_sentence = "A dog is running in the grass."
src_indices = preprocess_sentence(src_sentence, src_vocab)

# Get translation + attention weights
with torch.no_grad():
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(src_indices)]).to(device)
    
    # Encode
    encoder_outputs, hidden, cell = model_attention.encoder(src_tensor, src_len)
    
    # Decode step by step (collect attentions)
    attentions_list = []
    input = torch.LongTensor([tgt_vocab.sos_idx]).unsqueeze(0).to(device)
    
    for t in range(50):
        output, hidden, cell, attention = model_attention.decoder(
            input, hidden, cell, encoder_outputs
        )
        attentions_list.append(attention.cpu().numpy())
        
        pred_token = output.argmax(1).item()
        if pred_token == tgt_vocab.eos_idx:
            break
        input = torch.LongTensor([pred_token]).unsqueeze(0).to(device)

# Plot attention heatmap
attentions = np.vstack(attentions_list)

plt.figure(figsize=(10, 8))
sns.heatmap(attentions, cmap='Blues', cbar=True)
plt.xlabel('Source tokens')
plt.ylabel('Target tokens')
plt.title('Attention Weights Visualization')
plt.savefig('attention_heatmap.png', dpi=150)
plt.show()
```

---

## 🎯 Next Steps nếu chưa đạt 45%

Nếu sau khi train chỉ đạt ~42-43%, thêm:

### 1. Bidirectional Encoder (+2%)
```python
self.rnn = nn.LSTM(
    emb_dim, hid_dim, n_layers,
    bidirectional=True,  # Add this
    dropout=dropout
)
```

### 2. Beam Search (+2%)
- Implement beam_search() function
- Use beam_width=5

---

## 📝 Checklist

- [ ] Copy code vào notebook
- [ ] Run cells 1-5 theo thứ tự
- [ ] Đợi training hoàn thành (~2.5 giờ)
- [ ] Check BLEU score
- [ ] Nếu ≥45% → ✅ Done!
- [ ] Nếu <45% → Thêm Bidirectional Encoder

---

**Good luck! 🚀 Mục tiêu 45% trong tầm tay!**
