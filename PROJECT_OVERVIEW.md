# 📚 TỔNG QUAN ĐỒ ÁN NLP - DỊCH MÁY ANH-PHÁP

## 🎯 MỤC TIÊU BÀI TẬP

Xây dựng hệ thống **dịch máy tự động từ tiếng Anh sang tiếng Pháp** sử dụng mô hình **Encoder-Decoder LSTM** (không dùng Attention).

### Yêu cầu đầu ra:
- ✅ Model dịch được câu tiếng Anh → tiếng Pháp
- ✅ BLEU score trên tập test
- ✅ Mã nguồn chạy được từ đầu đến cuối trên Google Colab
- ✅ Báo cáo PDF đầy đủ (sơ đồ, biểu đồ, phân tích)
- ✅ Checkpoint file (.pth) để tái sử dụng model

---

## 📊 CẤU TRÚC BÀI TẬP (10 ĐIỂM)

| Task | Nội dung | Điểm | Trạng thái |
|------|----------|------|------------|
| **Task 1** | Thiết lập môi trường, config, utils | 3.0 | ✅ Hoàn thành |
| **Task 2** | Xử lý dữ liệu, DataLoader | 2.0 | ✅ Hoàn thành |
| **Task 3** | Xây dựng model Encoder-Decoder | 3.0 | ⏳ Chưa làm |
| **Task 4** | Training loop với early stopping | 1.5 | ⏳ Chưa làm |
| **Task 5** | Hàm translate() và đánh giá BLEU | 2.0 | ⏳ Chưa làm |
| **Task 6-8** | Phân tích lỗi, báo cáo, sơ đồ | 2.0 | ⏳ Chưa làm |

**Tổng điểm hiện tại: 5.0/13.5**

---

## 🗂️ DỮ LIỆU ĐẦU VÀO

### Dataset: Multi30K English-French
```
data/
├── train.en (29,000 câu tiếng Anh)
├── train.fr (29,000 câu tiếng Pháp tương ứng)
├── val.en   (1,014 câu validation)
├── val.fr   (1,014 câu validation)
├── test.en  (1,000 câu test)
└── test.fr  (1,000 câu test)
```

### Đặc điểm dữ liệu:
- **Parallel corpus**: Mỗi câu tiếng Anh có 1 câu tiếng Pháp tương ứng
- **Độ dài câu**: Trung bình 10-15 từ, tối đa 50 từ
- **Domain**: Mô tả hình ảnh (image captions)

---

## 🔄 QUY TRÌNH XỬLÝ - TỪ INPUT ĐẾN OUTPUT

### 📥 **BƯỚC 1: XỬ LÝ DỮ LIỆU THÔ**

**Input:** 
```
train.en: "A man is walking in the street."
train.fr: "Un homme marche dans la rue."
```

**Xử lý:**
1. **Tokenization** (tách từ):
   ```python
   EN: ["a", "man", "is", "walking", "in", "the", "street", "."]
   FR: ["un", "homme", "marche", "dans", "la", "rue", "."]
   ```

2. **Build Vocabulary** (tạo từ điển):
   - Đếm tần suất xuất hiện của từng từ
   - Lấy 10,000 từ phổ biến nhất
   - Tạo mapping: `word ↔ index`
   ```python
   "man" → 523
   "walking" → 1247
   ```

3. **Encode thành số**:
   ```python
   EN: [2, 523, 45, 1247, 89, 12, 678, 5, 3]  # 2=<sos>, 3=<eos>
   FR: [2, 312, 891, 456, 234, 67, 445, 5, 3]
   ```

**Output:** 
- `src_vocab`: Từ điển tiếng Anh (10,000 tokens)
- `tgt_vocab`: Từ điển tiếng Pháp (10,000 tokens)

---

### 📦 **BƯỚC 2: TẠO BATCHES**

**Input:** 29,000 câu đã encode

**Xử lý:**
1. **Chia thành batches** (64 câu/batch):
   ```
   Batch 1: 64 câu
   Batch 2: 64 câu
   ...
   Batch 454: 64 câu
   ```

2. **Sắp xếp theo độ dài giảm dần** (trong mỗi batch):
   ```
   Câu 1: 25 tokens (dài nhất)
   Câu 2: 23 tokens
   ...
   Câu 64: 8 tokens (ngắn nhất)
   ```

3. **Padding** (thêm <pad> cho câu ngắn):
   ```
   Câu 1: [2, 523, 45, ..., 3]           (25 tokens)
   Câu 64: [2, 89, 12, 3, 0, 0, 0, ...]  (8 tokens + 17 padding)
   ```

**Output:**
- `train_loader`: 454 batches
- `val_loader`: 16 batches
- `test_loader`: 16 batches

**Tại sao cần làm vậy?**
- Sắp xếp giảm dần → dùng `pack_padded_sequence` → LSTM xử lý nhanh hơn
- Padding → tất cả câu cùng độ dài → xử lý song song trên GPU

---

### 🧠 **BƯỚC 3: XÂY DỰNG MODEL**

#### Kiến trúc Encoder-Decoder:

```
INPUT (English)                    OUTPUT (French)
    ↓                                     ↓
["a", "man", "is", "walking"]    ["un", "homme", "marche"]
    ↓                                     ↓
[2, 523, 45, 1247, 3]           [2, 312, 891, 456, 3]
    ↓                                     ↑
┌─────────────────┐                      │
│    ENCODER      │                      │
│  (LSTM 2 layers)│                      │
│   Hidden: 512   │                      │
└────────┬────────┘                      │
         │ Context Vector                │
         │ (hidden + cell state)         │
         └──────────────────────┬────────┘
                                │
                         ┌──────▼──────┐
                         │   DECODER   │
                         │(LSTM 2 layers)│
                         │  Hidden: 512 │
                         └─────────────┘
```

**Encoder:**
- Đọc câu tiếng Anh từ trái → phải
- Mỗi từ → embedding vector (256 chiều)
- LSTM xử lý chuỗi → tạo context vector
- Context vector = tóm tắt toàn bộ câu tiếng Anh

**Decoder:**
- Nhận context vector từ Encoder
- Sinh từng từ tiếng Pháp từ trái → phải
- Mỗi bước sinh 1 từ dựa trên:
  - Context vector
  - Từ đã sinh trước đó

**Output:** Model đã khởi tạo, sẵn sàng training

---

### 🏋️ **BƯỚC 4: TRAINING (HỌC)**

**Input:** 
- Model chưa train (random weights)
- 454 batches dữ liệu training

**Quá trình 1 epoch:**
```
Epoch 1:
  Batch 1/454: Loss = 8.523
  Batch 2/454: Loss = 8.234
  ...
  Batch 454/454: Loss = 6.123
  → Train Loss = 7.123
  
  Validation:
  → Val Loss = 5.234
  
  ✅ Val loss giảm → Lưu model
```

**Các kỹ thuật quan trọng:**

1. **Teacher Forcing (50%)**:
   - 50% lần: Decoder nhận từ đúng từ ground truth
   - 50% lần: Decoder nhận từ dự đoán của chính nó
   - → Model học nhanh hơn, ổn định hơn

2. **Early Stopping (patience=3)**:
   ```
   Epoch 5: val_loss = 3.2 ✅ (best)
   Epoch 6: val_loss = 3.3 (tăng lần 1)
   Epoch 7: val_loss = 3.4 (tăng lần 2)
   Epoch 8: val_loss = 3.5 (tăng lần 3)
   → Dừng training! Tránh overfitting
   ```

3. **Learning Rate Scheduling**:
   ```
   Epoch 1-3: lr = 0.001
   Val loss không giảm sau 2 epochs
   → Epoch 4: lr = 0.0005 (giảm 50%)
   ```

4. **Gradient Clipping**:
   - Giới hạn gradient ≤ 1.0
   - Tránh exploding gradients

**Output:**
- `best_model.pth`: Model có val_loss thấp nhất
- `train_losses`, `val_losses`: Lịch sử loss để vẽ biểu đồ

---

### 🔍 **BƯỚC 5: DỊCH CÂU MỚI (INFERENCE)**

**Input:** Câu tiếng Anh mới
```
"The cat is sleeping on the bed."
```

**Quá trình dịch (Greedy Decoding):**

```
Step 0: Tokenize + Encode
  → [2, 12, 234, 67, 456, 89, 12, 890, 5, 3]

Step 1: Encoder xử lý
  → Context vector = [0.23, -0.45, 0.67, ...]

Step 2: Decoder bắt đầu với <sos>
  Input: <sos> → Output: "le" (xác suất cao nhất)

Step 3: Decoder nhận "le"
  Input: "le" → Output: "chat"

Step 4: Decoder nhận "chat"
  Input: "chat" → Output: "dort"

Step 5: Decoder nhận "dort"
  Input: "dort" → Output: "sur"

...

Step N: Decoder sinh <eos>
  → Dừng lại!
```

**Output:**
```
"le chat dort sur le lit ."
```

---

### 📊 **BƯỚC 6: ĐÁNH GIÁ (EVALUATION)**

#### BLEU Score:
Đo độ tương đồng giữa câu dịch và câu tham chiếu.

**Ví dụ:**
```
Reference: "le chat dort sur le lit ."
Predicted: "le chat est sur le lit ."

BLEU-1 (1-gram): 85.7% (6/7 từ trùng)
BLEU-2 (2-gram): 66.7% (4/6 cặp từ trùng)
BLEU-4 (4-gram): 50.0%

→ BLEU Score: 67.5% (trung bình có trọng số)
```

**Chạy trên toàn bộ test set:**
```python
for 1,000 câu trong test:
    translated = model.translate(câu_tiếng_Anh)
    bleu_score = compute_bleu(translated, câu_tham_chiếu)

→ Average BLEU = 28.5% (ví dụ)
```

**Benchmark:**
- BLEU < 20%: Kém
- BLEU 20-30%: Trung bình (model cơ bản)
- BLEU 30-40%: Khá
- BLEU > 40%: Tốt (cần Attention, Transformer)

---

## 🎯 CÁCH ĐẠT KẾT QUẢ CAO NHẤT

### ✅ **CẤP ĐỘ CƠ BẢN (7-8 điểm)**

1. **Hoàn thành đầy đủ Task 1-5**
2. **Hyperparameters mặc định:**
   - Batch size: 64
   - Embedding: 256
   - Hidden: 512
   - Layers: 2
   - Dropout: 0.3
   - Epochs: 15

3. **BLEU score:** 18-25%

### ⭐ **CẤP ĐỘ TỐT (8-9 điểm)**

1. **Tối ưu hyperparameters:**
   ```python
   BATCH_SIZE = 128          # Tăng batch size
   EMBEDDING_DIM = 512       # Tăng embedding
   HIDDEN_SIZE = 1024        # Tăng hidden size
   NUM_LAYERS = 3            # Thêm layer
   DROPOUT = 0.5             # Tăng dropout
   TEACHER_FORCING_RATIO = 0.7  # Tăng teacher forcing
   ```

2. **Kỹ thuật bổ sung:**
   - Learning rate decay
   - Gradient clipping = 1.0
   - Weight initialization (Xavier/He)

3. **BLEU score:** 25-30%

### 🏆 **CẤP ĐỘ XUẤT SẮC (9-10 điểm)**

1. **Thêm Attention Mechanism:**
   ```python
   class Attention(nn.Module):
       # Decoder chú ý đến từng từ của Encoder
       # Thay vì chỉ dùng context vector cố định
   ```

2. **Data Augmentation:**
   - Back-translation (dịch ngược lại)
   - Paraphrase (diễn đạt lại)

3. **Ensemble Models:**
   - Train 3-5 models khác nhau
   - Average predictions

4. **Beam Search (thay Greedy):**
   - Giữ top-5 candidates mỗi bước
   - Chọn sequence có xác suất cao nhất

5. **BLEU score:** 30-35%+

6. **Báo cáo chất lượng:**
   - Phân tích sâu 10-20 ví dụ lỗi
   - So sánh với baseline
   - Đề xuất cải tiến cụ thể
   - Vẽ sơ đồ kiến trúc đẹp
   - Biểu đồ loss, attention weights

---

## 📋 CHECKLIST HOÀN THÀNH

### Task 1: Môi trường (3.0đ) ✅
- [x] File `config.py` với tất cả hyperparameters
- [x] File `utils.py` với Vocabulary class
- [x] File `requirements.txt`
- [x] File `README.md`

### Task 2: Data Processing (2.0đ) ✅
- [x] Tokenization đơn giản (lowercase + regex)
- [x] Build vocabularies (10,000 tokens/ngôn ngữ)
- [x] Sắp xếp batch theo độ dài giảm dần
- [x] Padding sequences
- [x] DataLoader với batch size 32-128

### Task 3: Model (3.0đ) ⏳
- [ ] Class `Encoder` (LSTM 2 layers)
- [ ] Class `Decoder` (LSTM 2 layers)
- [ ] Class `Seq2Seq` (kết hợp Encoder-Decoder)
- [ ] Context vector từ hidden + cell state
- [ ] Teacher forcing trong training

### Task 4: Training (1.5đ) ⏳
- [ ] Hàm `train_epoch()`
- [ ] Hàm `evaluate()` (validation)
- [ ] Early stopping (patience=3)
- [ ] Learning rate scheduler
- [ ] Lưu checkpoint model tốt nhất
- [ ] Vẽ biểu đồ train/val loss

### Task 5: Evaluation (2.0đ) ⏳
- [ ] Hàm `translate()` với greedy decoding
- [ ] Tính BLEU score trên test set
- [ ] Test trên 5-10 câu mẫu
- [ ] So sánh với ground truth

### Task 6-8: Báo cáo (2.0đ) ⏳
- [ ] Sơ đồ kiến trúc model
- [ ] Biểu đồ train/val loss
- [ ] Bảng kết quả BLEU score
- [ ] Phân tích 5 ví dụ dịch sai
- [ ] Đề xuất cải tiến
- [ ] Export PDF

---

## 🚀 TIMELINE ĐỀ XUẤT

**Tổng thời gian: 5-7 ngày**

| Ngày | Công việc | Thời gian |
|------|-----------|-----------|
| Ngày 1 | Task 1-2: Setup + Data | 4-6 giờ |
| Ngày 2 | Task 3: Implement Model | 4-6 giờ |
| Ngày 3 | Task 4: Training (chạy overnight) | 8-12 giờ |
| Ngày 4 | Task 5: Evaluation + Debug | 3-4 giờ |
| Ngày 5 | Task 6-8: Báo cáo + Phân tích | 4-6 giờ |
| Ngày 6 | Review + Hoàn thiện | 2-3 giờ |
| Ngày 7 | Buffer (dự phòng) | - |

**Deadline:** 14/12/2025 (23:59)  
**Còn lại:** 7 ngày

---

## 🔥 LƯU Ý QUAN TRỌNG

### ⚠️ **Điểm dễ mất điểm:**

1. **Không sắp xếp batch** → pack_padded_sequence lỗi
2. **Quên padding** → tensor shape không đều
3. **Teacher forcing = 1.0** → model không học được tự sinh
4. **Không early stopping** → overfitting
5. **BLEU < 15%** → model học kém
6. **Báo cáo thiếu sơ đồ/biểu đồ** → mất điểm trình bày

### ✅ **Cách đảm bảo điểm cao:**

1. **Code chạy được từ đầu đến cuối** (quan trọng nhất!)
2. **Comment code rõ ràng** (giải thích từng bước)
3. **BLEU ≥ 20%** (chấp nhận được)
4. **Báo cáo đầy đủ:**
   - Giới thiệu bài toán
   - Sơ đồ kiến trúc
   - Kết quả (bảng, biểu đồ)
   - Phân tích lỗi (≥5 ví dụ)
   - Kết luận + đề xuất
5. **Nộp đủ file:**
   - PDF báo cáo
   - Notebook (.ipynb)
   - Checkpoint (.pth)

---

## 📁 CẤU TRÚC FOLDER CUỐI CÙNG

```
NLP_DO_AN/
├── data/                          # Dữ liệu
│   ├── train.en, train.fr
│   ├── val.en, val.fr
│   └── test.en, test.fr
│
├── src/                           # Source code
│   ├── config.py        ✅        # Đã có
│   ├── utils.py         ✅        # Đã có
│   ├── data_loader.py   ✅        # Đã có
│   ├── model.py         ⏳        # Cần làm
│   ├── train.py         ⏳        # Cần làm
│   └── evaluate.py      ⏳        # Cần làm
│
├── check_point/                   # Model weights
│   ├── best_model.pth   ⏳
│   ├── src_vocab.pth    ✅
│   └── tgt_vocab.pth    ✅
│
├── report/                        # Báo cáo
│   ├── PROGRESS_REPORT.md  ✅
│   ├── figures/          ⏳       # Sơ đồ, biểu đồ
│   └── final_report.pdf  ⏳
│
├── notebooks/                     # Notebook
│   └── NLP_Do_An.ipynb   ⏳
│
├── README.md             ✅
├── requirements.txt      ✅
├── COLAB_GUIDE.md        ✅
└── PROJECT_OVERVIEW.md   ✅ (file này)
```

---

## 🎓 TÀI LIỆU THAM KHẢO

1. **Paper gốc - Sequence to Sequence:**
   - Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks"

2. **Tutorial hay:**
   - PyTorch Seq2Seq Tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

3. **BLEU Score:**
   - Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"

---

**Good luck! 🚀**

*Tạo bởi: GitHub Copilot*  
*Ngày: 07/12/2025*
