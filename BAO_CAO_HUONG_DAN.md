# HƯỚNG DẪN VIẾT BÁO CÁO PDF - ĐỒ ÁN NLP CUỐI KÌ

## 🚨 QUY ĐỊNH CHÍNH THỨC (BẮT BUỘC)

### 1. Quy định chung:
- **Nhóm đồ án**: Tối đa **2 sinh viên**
- **Thời hạn nộp**: **14/12/2025 (23:59)** ⏰
- **Hình thức nộp**: **01 file PDF duy nhất** (báo cáo + mã nguồn trong phụ lục) qua hệ thống E-Learning
- **⚠️ KHÔNG chấp nhận nộp trễ** - 0 điểm nếu quá deadline

### 2. Yêu cầu về nội dung báo cáo PDF:
Báo cáo PDF **BẮT BUỘC** phải bao gồm:
- ✅ **Sơ đồ kiến trúc** (Encoder-Decoder architecture)
- ✅ **Biểu đồ train/val loss** (Training & Validation Loss curves)
- ✅ **BLEU score** (tính trên test set)
- ✅ **5 ví dụ dịch + phân tích** (Source → Prediction → Reference → Analysis)
- ✅ **Chương trình nguồn** (trong Phụ lục - có thể rút gọn hoặc highlight các phần chính)

### 3. Checkpoint mô hình:
- ✅ **`best_model.pth` BẮT BUỘC nộp** (đính kèm riêng hoặc link Google Drive trong báo cáo)
- File checkpoint dùng để kiểm tra kết quả training thực tế
- Nên include luôn `src_vocab.pth` và `tgt_vocab.pth`

### 4. Lưu ý quan trọng:
- 🚫 **KHÔNG sao chép mã** → 0 điểm nếu phát hiện
- ✅ **Mã nguồn phải chạy được** trên Google Colab hoặc máy local
- ✅ Báo cáo phải thể hiện sự hiểu biết về mô hình, không copy-paste từ internet

---

## 📋 YÊU CẦU ĐỊNH DẠNG

### Định dạng báo cáo:
- **Số trang**: 8-15 trang (không kể phụ lục mã nguồn)
- **Font chữ**: Times New Roman, 13pt cho nội dung chính, 11pt cho code
- **Lề**: Trái 3cm, Phải 2cm, Trên/Dưới 2.5cm
- **Dãn dòng**: 1.5 lines
- **Ngôn ngữ**: Tiếng Việt (có thể kèm thuật ngữ tiếng Anh)

---

## 📝 BỐ CỤC BÁO CÁO (THEO QUY ĐỊNH)

### TRANG BÌA
```
┌─────────────────────────────────────────────────────────────┐
│               ĐẠI HỌC QUỐC GIA TP.HCM                       │
│          TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN                │
│                                                             │
│                    [LOGO TRƯỜNG]                            │
│                                                             │
│                    BÁO CÁO ĐỒ ÁN                           │
│                   MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN              │
│                                                             │
│           ĐỀ TÀI: DỊCH MÁY ANH-PHÁP SỬ DỤNG               │
│             LSTM ENCODER-DECODER                            │
│                                                             │
│                                                             │
│  GVHD: [Tên giảng viên]                                    │
│  SVTH: [Họ và tên]                                         │
│  MSSV: [Mã số sinh viên]                                   │
│  Lớp: [Mã lớp]                                             │
│                                                             │
│            TP. Hồ Chí Minh, tháng 12/2025                  │
└─────────────────────────────────────────────────────────────┘
```

---

### MỤC LỤC (Trang 2)

```
MỤC LỤC

1. GIỚI THIỆU ............................................. 3
   1.1. Bối cảnh và động lực .............................. 3
   1.2. Mục tiêu đồ án .................................... 3
   1.3. Phạm vi và giới hạn ............................... 4

2. CÁC CÔNG TRÌNH LIÊN QUAN ................................ 5
   2.1. Lịch sử dịch máy .................................. 5
   2.2. Dịch máy neural (NMT) ............................. 5
   2.3. Encoder-Decoder với LSTM .......................... 6

3. PHƯƠNG PHÁP TIẾP CẬN .................................... 7
   3.1. Tổng quan kiến trúc (✅ SƠ ĐỒ KIẾN TRÚC) .......... 7
   3.2. Xử lý dữ liệu ..................................... 8
   3.3. Mô hình Encoder ................................... 9
   3.4. Mô hình Decoder ................................... 10
   3.5. Huấn luyện và tối ưu .............................. 11

4. THỰC NGHIỆM VÀ KẾT QUẢ .................................. 12
   4.1. Thiết lập thực nghiệm ............................. 12
   4.2. Kết quả huấn luyện (✅ BIỂU ĐỒ TRAIN/VAL LOSS) .... 13
   4.3. Đánh giá BLEU score (✅ BLEU SCORE) ............... 14
   4.4. 5 ví dụ dịch (✅ 5 VÍ DỤ + PHÂN TÍCH) ............. 15
   4.5. Phân tích lỗi và đề xuất cải tiến ................ 16

5. KẾT LUẬN ................................................ 17
   5.1. Tổng kết .......................................... 17
   5.2. Hạn chế của đề án ................................. 17
   5.3. Hướng phát triển tương lai ........................ 18

TÀI LIỆU THAM KHẢO ......................................... 19

PHỤ LỤC (✅ CHƯƠNG TRÌNH NGUỒN) ............................ 20
   A. Cấu hình và siêu tham số ............................ 20
   B. Code chính (Encoder, Decoder, Seq2Seq) .............. 21
   C. Code huấn luyện và inference ........................ 23
   D. Link checkpoint (best_model.pth) .................... 25
```

**📌 LƯU Ý:** Các phần đánh dấu ✅ là **BẮT BUỘC** theo quy định Section 11

---

## 📖 NỘI DUNG CHI TIẾT TỪNG PHẦN

### **1. GIỚI THIỆU (1-2 trang)**

#### 1.1. Bối cảnh và động lực
- Tầm quan trọng của dịch máy trong thời đại toàn cầu hóa
- Sự phát triển của Deep Learning trong NLP
- Ưu điểm của Neural Machine Translation so với Statistical MT

**Mẫu viết:**
```
Trong bối cảnh toàn cầu hóa, nhu cầu dịch thuật tự động ngày càng tăng cao. 
Dịch máy neural (Neural Machine Translation - NMT) đã chứng minh hiệu quả 
vượt trội so với các phương pháp thống kê truyền thống. Đồ án này tập trung 
vào việc xây dựng mô hình dịch Anh-Pháp sử dụng kiến trúc Encoder-Decoder 
với LSTM...
```

#### 1.2. Mục tiêu đồ án
- Xây dựng mô hình Seq2Seq với LSTM
- Đạt BLEU score ≥ 20% trên tập test
- Phân tích lỗi và đề xuất cải thiện

#### 1.3. Phạm vi và giới hạn
- Dataset: Multi30K (29,000 câu train)
- Cặp ngôn ngữ: Anh → Pháp
- Không sử dụng Attention (theo yêu cầu)

---

### **2. CÁC CÔNG TRÌNH LIÊN QUAN (1-2 trang)**

#### 2.1. Lịch sử dịch máy
- Rule-based MT (1950s-1990s)
- Statistical MT (1990s-2010s)
- Neural MT (2014-nay)

#### 2.2. Encoder-Decoder Framework
- **Sutskever et al. (2014)**: "Sequence to Sequence Learning with Neural Networks"
  - Kiến trúc Encoder-Decoder cơ bản
  - Sử dụng LSTM để xử lý chuỗi dài
  
- **Cho et al. (2014)**: RNN Encoder-Decoder
  - Giới thiệu GRU
  
- **Bahdanau et al. (2015)**: Attention Mechanism
  - Giải quyết vấn đề bottleneck của context vector

**Mẫu trích dẫn:**
```
Sutskever et al. [1] đã đề xuất kiến trúc Encoder-Decoder sử dụng LSTM, 
đạt BLEU 34.8 trên WMT'14 English-to-French. Mô hình này mã hóa toàn bộ 
câu nguồn thành một context vector cố định, sau đó decoder sinh ra câu đích...
```

---

### **3. PHƯƠNG PHÁP ĐỀ XUẤT (3-4 trang)**

#### 3.1. Kiến trúc tổng thể

**Sơ đồ kiến trúc (VẼ HÌNH):**
```
Input: "A dog is running"
   ↓
┌─────────────────────────────────────┐
│  TOKENIZATION & EMBEDDING           │
│  ["<sos>", "a", "dog", "is",        │
│   "running", "<eos>"]               │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│         ENCODER (LSTM)              │
│  • 2 layers, hidden=512             │
│  • Bidirectional: No (theo yêu cầu) │
│  • Dropout: 0.3                     │
│                                     │
│  Output: Context Vector (h_n, c_n) │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│         DECODER (LSTM)              │
│  • 2 layers, hidden=512             │
│  • Teacher forcing ratio: 0.5       │
│  • Output vocab size: 10,000        │
│                                     │
│  Output: ["un", "chien", "court"]  │
└─────────────────────────────────────┘
   ↓
Output: "Un chien court"
```

#### 3.2. Xử lý dữ liệu

**Bảng thống kê dataset:**
| Tập dữ liệu | Số câu | Độ dài TB (EN) | Độ dài TB (FR) |
|-------------|--------|----------------|----------------|
| Train       | 29,000 | 13.2 tokens    | 14.8 tokens    |
| Validation  | 1,014  | 13.5 tokens    | 15.1 tokens    |
| Test        | 1,000  | 12.8 tokens    | 14.3 tokens    |

**Các bước tiền xử lý:**
1. Tokenization: Regex-based, lowercase
2. Vocabulary: Top 10,000 từ phổ biến nhất
3. Special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>`
4. Padding & Packing: `pack_padded_sequence` để tối ưu

#### 3.3. Mô hình Encoder

**Công thức toán học:**
```
h_t, c_t = LSTM(emb(x_t), (h_{t-1}, c_{t-1}))

Trong đó:
- x_t: token thứ t của câu nguồn
- emb(): embedding layer (256 chiều)
- h_t: hidden state tại time step t
- c_t: cell state tại time step t
```

**Pseudo-code:**
```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
    
    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = pack_padded_sequence(embedded, src_len)
        outputs, (hidden, cell) = self.lstm(packed)
        return hidden, cell  # Context vector
```

#### 3.4. Mô hình Decoder

**Teacher Forcing:**
```
Teacher Forcing (ratio=0.5):
- 50% thời gian: dùng ground truth làm input
- 50% thời gian: dùng prediction của model làm input

Ví dụ:
Ground truth: "<sos> un chien court <eos>"
t=1: input="<sos>" → predict="un" ✓
t=2: input="un" (ground truth) → predict="chien" ✓
t=3: input="chien" (prediction) → predict="court" ✓
```

**Công thức:**
```
h_t, c_t = LSTM(emb(y_{t-1}), (h_{t-1}, c_{t-1}))
output_t = Linear(h_t)
pred_t = Softmax(output_t)
```

#### 3.5. Huấn luyện

**Hàm loss:**
```
Loss = CrossEntropyLoss(ignore_index=PAD_IDX)

L = -∑_{t=1}^{T} log P(y_t | y_{<t}, x)

Trong đó:
- y_t: token đúng tại vị trí t
- P(y_t | y_{<t}, x): xác suất dự đoán
```

**Cấu hình training:**
- Optimizer: Adam (lr=0.001, betas=(0.9, 0.999))
- Batch size: 64
- Epochs: 15 (với early stopping patience=3)
- Gradient clipping: max_norm=1
- Device: GPU Tesla T4 (Google Colab)

**Early Stopping:**
```
if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    save_checkpoint()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 3:
        break  # Dừng training
```

---

### **4. THỰC NGHIỆM VÀ KẾT QUẢ (3-4 trang)**

#### 4.1. Thiết lập thực nghiệm

**Bảng siêu tham số:**
| Tham số | Giá trị | Lý do chọn |
|---------|---------|------------|
| Embedding dim | 256 | Cân bằng giữa biểu diễn và tốc độ |
| Hidden dim | 512 | Đủ lớn để học phụ thuộc dài |
| Num layers | 2 | Tránh overfitting với dataset nhỏ |
| Dropout | 0.3 | Regularization |
| Batch size | 64 | Tối ưu cho GPU T4 |
| Learning rate | 0.001 | Giá trị chuẩn cho Adam |

#### 4.2. Kết quả huấn luyện (✅ BIỂU ĐỒ TRAIN/VAL LOSS)

**📊 Biểu đồ Loss (BẮT BUỘC):**
```
[CHÈN HÌNH từ notebook: training_validation_loss.png]

Mô tả chi tiết:
- Trục X: Epochs (1-15)
- Trục Y: Loss (0-5)
- Đường xanh (train): Giảm từ 4.2 → 1.8
- Đường đỏ (validation): Giảm từ 3.9 → 2.1
- Early stopping: Kích hoạt tại epoch 12
- Gap train-val: ~0.3 (overfitting nhẹ, chấp nhận được)
```

**Bảng kết quả training chi tiết:**
| Epoch | Train Loss | Val Loss | Train PPL | Val PPL | Time | Note |
|-------|------------|----------|-----------|---------|------|------|
| 1     | 4.256      | 3.892    | 70.45     | 49.12   | 8m   | Khởi đầu |
| 5     | 2.341      | 2.567    | 10.39     | 13.03   | 8m   | Giảm nhanh |
| 10    | 1.923      | 2.234    | 6.84      | 9.34    | 8m   | Ổn định |
| 12    | 1.812      | 2.156    | 6.12      | 8.64    | 8m   | Best model ✅ |

**Phân tích:**
- Model hội tụ tốt sau 12 epochs
- Validation loss giảm đều → không bị overfitting nghiêm trọng
- Perplexity giảm từ 70 → 6 cho train, 49 → 8.6 cho val
- Thời gian training: ~1.5 giờ trên GPU T4

---

#### 4.3. Đánh giá BLEU score (✅ BLEU SCORE)

**📈 BLEU Score trên Test Set (BẮT BUỘC):**
```
BLEU Score: 23.4%
Corpus size: 1,000 câu test
Smoothing: SmoothingFunction().method1

Đánh giá chi tiết:
✓ Đạt yêu cầu: ≥ 20% (theo đề bài)
✓ So với baseline (random): ~0% → Cải thiện 23.4%
✓ So với no-training: ~5% → Cải thiện 18.4%
✓ So với SOTA (Transformer + Attention): ~42% → Gap 18.6%
```

**Phân phối BLEU score:**
| BLEU Range | Số câu | Tỉ lệ | Đánh giá |
|------------|--------|-------|----------|
| ≥ 40% (Tốt) | 180 câu | 18% | Dịch chính xác |
| 20-40% (Khá) | 420 câu | 42% | Dịch chấp nhận được |
| 10-20% (Trung bình) | 250 câu | 25% | Còn nhiều lỗi |
| < 10% (Kém) | 150 câu | 15% | Dịch sai hoàn toàn |

---

#### 4.4. 5 ví dụ dịch + phân tích (✅ 5 VÍ DỤ BẮT BUỘC)

**📝 Ví dụ 1: Dịch chính xác (BLEU = 100%)**
```
Source (EN):     A dog is running in the grass
Prediction (FR): un chien court dans l'herbe
Reference (FR):  un chien court dans l'herbe
BLEU Score:      100.0%

✅ Phân tích:
- Dịch chính xác 100%, từng từ đều đúng
- Thứ tự từ đúng: "un chien" (a dog), "court" (is running), "dans l'herbe" (in the grass)
- Không có từ <unk>, tất cả từ đều trong vocabulary
- Câu đơn giản (7 từ) → Model xử lý tốt
```

**📝 Ví dụ 2: Dịch tốt nhưng từ đồng nghĩa (BLEU = 75%)**
```
Source (EN):     Two children playing soccer
Prediction (FR): deux enfants jouent au football
Reference (FR):  deux enfants jouent au foot
BLEU Score:      75.3%

✅ Phân tích:
- Dịch đúng nghĩa nhưng dùng "football" thay vì "foot"
- "football" = "foot" (từ đồng nghĩa) → cả 2 đều đúng
- Cấu trúc câu chính xác: "deux enfants jouent au..."
- BLEU giảm do không match exact string với reference
- Trong thực tế: đây là bản dịch CHÍNH XÁC
```

**📝 Ví dụ 3: Lỗi thứ tự từ (BLEU = 35.7%)**
```
Source (EN):     A red car on the road
Prediction (FR): une voiture sur la route rouge
Reference (FR):  une voiture rouge sur la route
BLEU Score:      35.7%

❌ Phân tích:
- Lỗi: "rouge" (red) đặt sai vị trí
- Model dịch: "une voiture sur la route rouge" (a car on the red road)
- Đúng phải: "une voiture rouge sur la route" (a red car on the road)
- Nguyên nhân: Tính từ trong tiếng Pháp thường đứng SAU danh từ
- Giải pháp: Thêm attention để học vị trí tính từ chính xác hơn
```

**📝 Ví dụ 4: Lỗi OOV - từ không có trong vocab (BLEU = 12.5%)**
```
Source (EN):     A motorcyclist is racing down the track
Prediction (FR): un <unk> est en train de <unk> sur la piste
Reference (FR):  un motocycliste fait de la course sur la piste
BLEU Score:      12.5%

❌ Phân tích:
- Lỗi nghiêm trọng: 2 từ <unk> (unknown)
- "motorcyclist" không có trong vocab 10,000 từ
- "racing" bị hiểu sai → dịch thành <unk>
- Chỉ dịch đúng: "un ... sur la piste" (on the track)
- Giải pháp:
  1. Tăng vocab size: 10K → 30K
  2. Dùng BPE: "motorcyclist" → ["motor", "cycl", "ist"]
```

**📝 Ví dụ 5: Lỗi câu dài - mất thông tin (BLEU = 18.2%)**
```
Source (EN):     A group of people are sitting on the beach watching the sunset
Prediction (FR): un groupe de personnes sont <unk> sur la plage
Reference (FR):  un groupe de personnes sont assis sur la plage regardant le coucher du soleil
BLEU Score:      18.2%

❌ Phân tích:
- Câu gốc dài: 13 từ
- Model chỉ dịch được nửa đầu: "un groupe de personnes sont ... sur la plage"
- Thiếu: "assis" (sitting), "regardant le coucher du soleil" (watching the sunset)
- Nguyên nhân: Context vector cố định 512-dim không đủ lưu thông tin
- Giải pháp:
  1. Attention mechanism: Focus vào từng phần của câu nguồn
  2. Tăng hidden_dim: 512 → 1024
  3. Bidirectional encoder: Đọc câu từ 2 chiều
```

**📊 Tổng kết 5 ví dụ:**
| Ví dụ | BLEU | Loại lỗi | Mức độ nghiêm trọng |
|-------|------|----------|---------------------|
| 1     | 100% | Không lỗi | ✅ Hoàn hảo |
| 2     | 75%  | Từ đồng nghĩa | ✅ Chấp nhận được |
| 3     | 36%  | Thứ tự từ | ⚠️ Cần cải thiện |
| 4     | 13%  | OOV (<unk>) | ❌ Lỗi nghiêm trọng |
| 5     | 18%  | Câu dài | ❌ Lỗi nghiêm trọng |

---

#### 4.5. Phân tích lỗi tổng quát và đề xuất cải tiến

**4 loại lỗi chính:**

**1. Câu dài (>15 từ) - 35% lỗi:**
```
Source: A group of people are sitting on the beach watching the sunset
Pred:   un groupe de personnes sont <unk> sur la plage
Ref:    un groupe de personnes sont assis sur la plage regardant le coucher du soleil
BLEU:   18.2%

Nguyên nhân: Context vector cố định không đủ để lưu thông tin câu dài
Giải pháp: Sử dụng Attention mechanism
```

**2. Từ OOV (<unk>) - 28% lỗi:**
```
Source: A motorcyclist is racing down the track
Pred:   un <unk> est en train de <unk> sur la piste
Ref:    un motocycliste fait de la course sur la piste
BLEU:   12.5%

Nguyên nhân: Từ "motorcyclist" không có trong vocab 10K
Giải pháp: Tăng vocab hoặc dùng subword (BPE)
```

**3. Lỗi ngữ pháp - 22% lỗi:**
```
Source: The dog is barking loudly
Pred:   le chien est aboie fort
Ref:    le chien aboie fort
BLEU:   25.3%

Nguyên nhân: Dùng cả "est" và "aboie" (thừa trợ động từ)
Giải pháp: Tăng dữ liệu training, cải thiện model
```

**4. Thứ tự từ sai - 15% lỗi:**
```
Source: A red car on the road
Pred:   une voiture sur la route rouge
Ref:    une voiture rouge sur la route
BLEU:   35.7%

Nguyên nhân: "rouge" nên đứng sau "voiture" chứ không phải "route"
Giải pháp: Học attention để hiểu cấu trúc câu tốt hơn
```

**Biểu đồ phân bố lỗi:**
```
[CHÈN HÌNH: error_distribution.png]

Câu dài (>15 từ):     ████████████████████████████ 35%
OOV (<unk>):          ███████████████████████ 28%
Ngữ pháp:             ███████████████ 22%
Thứ tự từ:            ██████ 15%
```

---

### **5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (1-2 trang)**

#### 5.1. Kết luận

**Đóng góp chính:**
1. ✅ Xây dựng thành công mô hình Seq2Seq với LSTM
2. ✅ Đạt BLEU 23.4% (vượt mục tiêu 20%)
3. ✅ Phân tích chi tiết 4 loại lỗi phổ biến
4. ✅ Đề xuất 5 hướng cải thiện cụ thể

**So sánh với yêu cầu:**
| Yêu cầu | Kết quả | Điểm |
|---------|---------|------|
| Cài đặt Encoder-Decoder | ✅ Hoàn thành | 3.0/3.0 |
| Xử lý dữ liệu (DataLoader) | ✅ Hoàn thành | 2.0/2.0 |
| Training + Early stopping | ✅ Hoàn thành | 1.5/1.5 |
| Hàm translate() | ✅ Hoàn thành | 1.0/1.0 |
| BLEU score + plots | ✅ Hoàn thành | 1.0/1.0 |
| Error analysis | ✅ Hoàn thành | 1.0/1.0 |
| Code quality | ✅ Hoàn thành | 0.5/0.5 |
| Báo cáo | ✅ Hoàn thành | 0.5/0.5 |
| **TỔNG** | | **10.0/10.0** |

#### 5.2. Hạn chế

1. **Context vector cố định**: Không thể lưu đủ thông tin cho câu dài
2. **Vocab hạn chế**: 10K từ → nhiều OOV
3. **Không có Attention**: Không thể focus vào từ quan trọng
4. **Dataset nhỏ**: 29K câu so với 4.5M của WMT'14

#### 5.3. Hướng phát triển

**5 cải tiến đề xuất (theo thứ tự ưu tiên):**

**1. Attention Mechanism (+10-15% BLEU):**
```
Luong Attention:
score(h_t, h_s) = h_t^T W h_s
α_t = softmax(score(h_t, h_s))
context_t = ∑ α_t * h_s

Ước tính: BLEU 23% → 33-38%
```

**2. Subword Tokenization (BPE) (+3-5% BLEU):**
```
Ví dụ BPE:
"motorcyclist" → ["motor", "cycl", "ist"]
"photographie" → ["photo", "graph", "ie"]

Ưu điểm: Giảm OOV từ 28% → 5%
Ước tính: BLEU 23% → 26-28%
```

**3. Beam Search (+2-4% BLEU):**
```python
def beam_search(model, src, beam_width=5):
    # Thay vì chọn 1 best token (greedy)
    # Giữ top-K candidates tại mỗi step
    # Chọn sequence có tổng score cao nhất
```

**4. Tăng dữ liệu (WMT 2014) (+5-10% BLEU):**
```
WMT 2014 English-French:
- 4.5M câu train (vs 29K hiện tại)
- Đa dạng domain (news, web, parliament)

Ước tính: BLEU 23% → 28-33%
```

**5. Scheduled Sampling (+1-2% BLEU):**
```
Giảm dần teacher forcing ratio:
Epoch 1-5:   ratio = 0.8
Epoch 6-10:  ratio = 0.5
Epoch 11+:   ratio = 0.2

Giúp model ổn định hơn khi inference
```

**Roadmap cải thiện:**
```
Giai đoạn 1 (2 tuần): Attention → 33-38% BLEU
Giai đoạn 2 (1 tuần): BPE → 36-41% BLEU
Giai đoạn 3 (3 ngày): Beam search → 38-45% BLEU
Giai đoạn 4 (1 tuần): WMT 2014 → 43-50% BLEU

Mục tiêu cuối: BLEU ≥ 40% (gần Transformer baseline)
```

---

## 📚 TÀI LIỆU THAM KHẢO

**Định dạng IEEE:**

```
[1] I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to sequence learning 
    with neural networks," in Advances in neural information processing 
    systems, 2014, pp. 3104-3112.

[2] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by 
    jointly learning to align and translate," arXiv preprint arXiv:1409.0473, 
    2014.

[3] M.-T. Luong, H. Pham, and C. D. Manning, "Effective approaches to 
    attention-based neural machine translation," arXiv preprint 
    arXiv:1508.04025, 2015.

[4] R. Sennrich, B. Haddow, and A. Birch, "Neural machine translation of 
    rare words with subword units," in Proceedings of ACL, 2016, pp. 1715-1725.

[5] A. Vaswani et al., "Attention is all you need," in Advances in neural 
    information processing systems, 2017, pp. 5998-6008.

[6] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: a method for 
    automatic evaluation of machine translation," in Proceedings of ACL, 
    2002, pp. 311-318.

[7] P. Koehn, "Statistical machine translation," Cambridge University Press, 
    2010.

[8] Y. Wu et al., "Google's neural machine translation system: Bridging the 
    gap between human and machine translation," arXiv preprint 
    arXiv:1609.08144, 2016.
```

---

## 📎 PHỤ LỤC (✅ CHƯƠNG TRÌNH NGUỒN - BẮT BUỘC)

**⚠️ LƯU Ý QUAN TRỌNG:**
- Phụ lục phải chứa **CHƯƠNG TRÌNH NGUỒN** (code Python)
- Có thể rút gọn code nhưng phải bao gồm các phần chính
- Highlight các đoạn code quan trọng (Encoder, Decoder, Training loop)
- Nếu code quá dài (>2,000 dòng), chỉ include các phần core và note "Full code: [Link GitHub]"

---

### Phụ lục A: Cấu hình và siêu tham số

**Bảng cấu hình đầy đủ:**

```python
# ============================================
# CONFIGURATION - NLP FINAL PROJECT
# English-French Machine Translation
# ============================================

CONFIG = {
    # ===== DATA CONFIGURATION =====
    'max_vocab_size': 10000,      # Top 10K từ phổ biến nhất
    'max_seq_len': 50,            # Độ dài tối đa của câu
    'min_freq': 2,                # Bỏ từ xuất hiện < 2 lần
    
    # ===== MODEL ARCHITECTURE =====
    'emb_dim': 256,               # Embedding dimension
    'hid_dim': 512,               # LSTM hidden dimension
    'n_layers': 2,                # Số lớp LSTM
    'dropout': 0.3,               # Dropout ratio (regularization)
    
    # ===== TRAINING CONFIGURATION =====
    'batch_size': 64,             # Batch size (tối ưu cho T4 GPU)
    'num_epochs': 15,             # Số epochs (với early stopping)
    'learning_rate': 0.001,       # Learning rate cho Adam
    'clip': 1.0,                  # Gradient clipping max_norm
    'teacher_forcing_ratio': 0.5, # Teacher forcing probability
    'early_stopping_patience': 3, # Dừng sau 3 epochs không cải thiện
    
    # ===== SPECIAL TOKENS =====
    'pad_token': '<pad>',         # Padding token (idx=0)
    'unk_token': '<unk>',         # Unknown token (idx=1)
    'sos_token': '<sos>',         # Start of sequence (idx=2)
    'eos_token': '<eos>',         # End of sequence (idx=3)
    
    # ===== DEVICE =====
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Đường dẫn dữ liệu
DATA_PATHS = {
    'train_en': 'data/train.en',
    'train_fr': 'data/train.fr',
    'val_en': 'data/val.en',
    'val_fr': 'data/val.fr',
    'test_en': 'data/test.en',
    'test_fr': 'data/test.fr',
}

# Checkpoint paths
CHECKPOINT_PATHS = {
    'best_model': 'check_point/best_model.pth',
    'src_vocab': 'check_point/src_vocab.pth',
    'tgt_vocab': 'check_point/tgt_vocab.pth',
}
```

---

### Phụ lục B: Code chính (✅ CORE IMPLEMENTATION)

#### B.1. Vocabulary Class

```python
class Vocabulary:
    """
    Quản lý từ điển cho source/target language
    """
    def __init__(self, max_size=10000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.word_freq = {}
    
    def build_vocab(self, sentences):
        """Xây dựng vocabulary từ danh sách câu"""
        # Đếm tần suất
        for sent in sentences:
            for word in sent:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Lọc từ theo min_freq và max_size
        valid_words = sorted(
            [(w, f) for w, f in self.word_freq.items() if f >= self.min_freq],
            key=lambda x: x[1], reverse=True
        )[:self.max_size - 4]  # Trừ 4 special tokens
        
        # Thêm vào vocab
        for word, _ in valid_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, tokens):
        """Chuyển list tokens → list indices"""
        return [self.word2idx.get(w, 1) for w in tokens]  # 1 = <unk>
    
    def decode(self, indices):
        """Chuyển list indices → list tokens"""
        return [self.idx2word.get(i, '<unk>') for i in indices]
```

#### B.2. Encoder Class

```python
class Encoder(nn.Module):
    """
    LSTM Encoder: Mã hóa câu nguồn thành context vector
    """
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=False
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        """
        Args:
            src: [src_len, batch_size] - Câu nguồn
            src_len: [batch_size] - Độ dài thực của mỗi câu
        Returns:
            hidden: [n_layers, batch_size, hid_dim]
            cell:   [n_layers, batch_size, hid_dim]
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))  # [src_len, batch, emb_dim]
        
        # Pack padded sequence (tối ưu LSTM)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), enforce_sorted=True
        )
        
        # LSTM forward
        packed_outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack (nếu cần dùng outputs)
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        return hidden, cell  # Context vector
```

#### B.3. Decoder Class

```python
class Decoder(nn.Module):
    """
    LSTM Decoder với Teacher Forcing
    """
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=False
        )
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        """
        Args:
            input: [batch_size] - Token hiện tại
            hidden: [n_layers, batch_size, hid_dim]
            cell:   [n_layers, batch_size, hid_dim]
        Returns:
            prediction: [batch_size, vocab_size] - Xác suất cho mỗi token
            hidden, cell: Context mới
        """
        # input: [batch] → [1, batch]
        input = input.unsqueeze(0)
        
        # Embedding
        embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [1, batch, hid_dim]
        
        # Linear projection
        prediction = self.fc_out(output.squeeze(0))  # [batch, vocab_size]
        
        return prediction, hidden, cell
```

#### B.4. Seq2Seq Model

```python
class Seq2Seq(nn.Module):
    """
    Seq2Seq Model = Encoder + Decoder
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Kiểm tra hidden_dim phải giống nhau
        assert encoder.lstm.hidden_size == decoder.lstm.hidden_size, \
            "Encoder và Decoder phải có cùng hidden_dim!"
    
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [src_len, batch_size]
            src_len: [batch_size]
            trg: [trg_len, batch_size]
            teacher_forcing_ratio: float (0-1)
        Returns:
            outputs: [trg_len, batch_size, vocab_size]
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
        
        # Tensor lưu outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # ENCODE
        hidden, cell = self.encoder(src, src_len)
        
        # DECODE
        input = trg[0, :]  # <sos> token
        
        for t in range(1, trg_len):
            # Decoder forward
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # Greedy: chọn token có prob cao nhất
            
            input = trg[t] if teacher_force else top1
        
        return outputs
```

#### B.5. Training Loop (Rút gọn)

```python
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src, src_len = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        # Forward
        output = model(src, src_len, trg)
        
        # Tính loss (bỏ <sos> token)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)
```

#### B.6. Translate Function (Inference)

```python
def translate(sentence, model, src_vocab, tgt_vocab, device, max_len=50):
    """
    Dịch 1 câu tiếng Anh sang tiếng Pháp
    """
    model.eval()
    
    # Tokenize
    tokens = tokenize_sentence(sentence, language="en")
    tokens = ['<sos>'] + tokens + ['<eos>']
    
    # Encode
    src_indexes = src_vocab.encode(tokens)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)])
    
    # Encoder
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    
    # Decoder (Greedy)
    trg_indexes = [tgt_vocab.word2idx['<sos>']]
    
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == tgt_vocab.word2idx['<eos>']:
            break
    
    # Decode
    trg_tokens = tgt_vocab.decode(trg_indexes)
    return ' '.join(trg_tokens[1:-1])  # Bỏ <sos> và <eos>
```

---

### Phụ lục C: Link Checkpoint và Code đầy đủ

**🔗 Google Drive Links:**
```
1. Checkpoint files:
   - best_model.pth:  [Link Google Drive]
   - src_vocab.pth:   [Link Google Drive]
   - tgt_vocab.pth:   [Link Google Drive]

2. Full notebook:
   - NLP_Do_An_EnFr_Translation.ipynb: [Link Google Drive hoặc GitHub]

3. GitHub Repository:
   - https://github.com/[username]/NLP_CUOIKY
```

**⚠️ Cách nộp checkpoint:**
- Option 1: Đính kèm trực tiếp file .pth vào E-Learning (nếu < 100MB)
- Option 2: Upload lên Google Drive, include link trong báo cáo PDF
- Option 3: Upload lên GitHub repository, include link trong báo cáo

---

## 🎨 MẸO THIẾT KẾ BÁO CÁO ĐẸP

### 1. Màu sắc và định dạng

**Sử dụng màu cho:**
- Header các section: Màu xanh dương (#2E86AB)
- Highlight code: Background xám nhạt (#F5F5F5)
- Chú thích hình: Màu xám đậm (#666666)
- Link tham khảo: Màu xanh (#0066CC)

### 2. Hình ảnh và sơ đồ

**Phải có ít nhất:**
- 1 sơ đồ kiến trúc tổng thể (trang 7)
- 2 biểu đồ loss (trang 13)
- 1 biểu đồ phân bố lỗi (trang 15)
- 5-10 ảnh minh họa khác (công thức, bảng, flowchart)

**Tool vẽ sơ đồ:**
- **draw.io**: Miễn phí, vẽ architecture
- **Matplotlib**: Vẽ biểu đồ loss từ notebook
- **LaTeX TikZ**: Vẽ công thức toán đẹp

### 3. Bảng biểu

**Format chuẩn:**
```
┌────────────────────────────────────────────────────┐
│ Header 1      │ Header 2     │ Header 3          │
├────────────────────────────────────────────────────┤
│ Data 1        │ Data 2       │ Data 3            │
│ Data 4        │ Data 5       │ Data 6            │
└────────────────────────────────────────────────────┘
```

- Border: Thin line (0.5pt)
- Header: Bold, background xám nhạt
- Alignment: Số (phải), Text (trái)

### 4. Code trong báo cáo

**Format code block:**
```python
# Font: Courier New 11pt
# Background: #F8F8F8
# Border: 1pt solid #DDDDDD
# Padding: 10px

def example_function():
    """Docstring"""
    return result
```

---

## ✅ CHECKLIST TRƯỚC KHI NỘP (QUAN TRỌNG!)

### 🚨 YÊU CẦU BẮT BUỘC (theo Section 11):

**1. Mã nguồn phải chạy được:**
- [ ] ✅ Mã nguồn chạy được trên Google Colab hoặc máy local
- [ ] ✅ Đã test lại toàn bộ notebook từ đầu (Runtime → Restart and run all)
- [ ] ✅ Không có lỗi khi chạy (ngoại trừ warnings không ảnh hưởng)
- [ ] ✅ Checkpoint `best_model.pth` được tạo thành công

**2. Báo cáo PDF phải bao gồm (5 yêu cầu bắt buộc):**
- [ ] ✅ **Sơ đồ kiến trúc** (Encoder-Decoder architecture) - có trong Section 3.1
- [ ] ✅ **Biểu đồ train/val loss** (Training & Validation Loss curves) - có trong Section 4.2
- [ ] ✅ **BLEU score** (tính trên test set) - có trong Section 4.3
- [ ] ✅ **5 ví dụ dịch + phân tích** (Source → Prediction → Reference → Analysis) - có trong Section 4.4
- [ ] ✅ **Chương trình nguồn** (trong Phụ lục) - có trong Phụ lục B, C

**3. Checkpoint mô hình:**
- [ ] ✅ File `best_model.pth` đã được tạo
- [ ] ✅ File `src_vocab.pth` đã được tạo
- [ ] ✅ File `tgt_vocab.pth` đã được tạo
- [ ] ✅ Đã upload checkpoint lên Google Drive (nếu file quá lớn)
- [ ] ✅ Link Google Drive được include trong báo cáo (Phụ lục C)

**4. Tính trung thực:**
- [ ] ⚠️ **KHÔNG sao chép mã nguồn** từ internet/bạn bè → 0 điểm nếu phát hiện
- [ ] ✅ Code có comment bằng tiếng Việt để thể hiện sự hiểu biết
- [ ] ✅ Báo cáo viết bằng ngôn ngữ của bản thân, không copy-paste

---

### 📋 CHECKLIST NỘI DUNG:

**Trang bìa & Mục lục:**
- [ ] Trang bìa đầy đủ: Tên trường, đề tài, GVHD, SVTH, MSSV, lớp, ngày tháng
- [ ] Mục lục có số trang chính xác
- [ ] Các section được đánh số đúng (1, 2, 3, 4, 5)

**Phần chính (Section 1-5):**
- [ ] Section 1: Giới thiệu (bối cảnh, mục tiêu, phạm vi)
- [ ] Section 2: Các công trình liên quan (≥3 papers, cite đúng)
- [ ] Section 3: Phương pháp (✅ sơ đồ kiến trúc, công thức toán, pseudo-code)
- [ ] Section 4.2: ✅ Biểu đồ train/val loss (có hình ảnh)
- [ ] Section 4.3: ✅ BLEU score (có con số cụ thể, ví dụ: 23.4%)
- [ ] Section 4.4: ✅ 5 ví dụ dịch (mỗi ví dụ có: Source, Prediction, Reference, BLEU, Phân tích)
- [ ] Section 5: Kết luận (tổng kết, hạn chế, hướng phát triển)

**Hình ảnh & Bảng biểu:**
- [ ] Tất cả hình ảnh có caption (Hình 1: ..., Hình 2: ...)
- [ ] Tất cả hình ảnh được reference trong text (xem Hình 1, ...)
- [ ] Tất cả bảng biểu có tiêu đề (Bảng 1: ..., Bảng 2: ...)
- [ ] Biểu đồ loss rõ ràng, có trục x/y, legend
- [ ] Sơ đồ kiến trúc dễ hiểu, có chú thích

**Tài liệu tham khảo:**
- [ ] Có ≥ 5 nguồn tham khảo
- [ ] Cite đúng format IEEE (hoặc ACL)
- [ ] Tất cả citation trong text đều có trong References
- [ ] Papers chính: Sutskever 2014, Bahdanau 2015, Luong 2015

**Phụ lục (✅ Chương trình nguồn):**
- [ ] ✅ Phụ lục A: Cấu hình & siêu tham số
- [ ] ✅ Phụ lục B: Code chính (Encoder, Decoder, Seq2Seq, Training, Inference)
- [ ] ✅ Phụ lục C: Link checkpoint và code đầy đủ (Google Drive hoặc GitHub)
- [ ] Code có comment, indent đúng, dễ đọc

---

### 📐 CHECKLIST ĐỊNH DẠNG:

**Font & Spacing:**
- [ ] Font Times New Roman 13pt (nội dung chính)
- [ ] Font 11pt hoặc Courier New cho code
- [ ] Dãn dòng 1.5 lines
- [ ] Lề: Trái 3cm, Phải 2cm, Trên 2.5cm, Dưới 2.5cm

**Số trang & Header:**
- [ ] Số trang ở cuối trang, giữa (bắt đầu từ trang Giới thiệu)
- [ ] Trang bìa không đánh số
- [ ] Mục lục không đánh số (hoặc đánh số La Mã: i, ii, iii)

**Chất lượng:**
- [ ] Không có lỗi chính tả
- [ ] Không có lỗi ngữ pháp
- [ ] Câu văn mạch lạc, rõ ràng
- [ ] Thuật ngữ tiếng Anh được in nghiêng (Machine Translation, BLEU score)

---

### 💻 CHECKLIST KỸ THUẬT:

**Code chất lượng:**
- [ ] Tất cả function có docstring
- [ ] Code có comment giải thích logic (bằng tiếng Việt)
- [ ] Naming convention rõ ràng (src_vocab, tgt_vocab, không dùng v1, v2)
- [ ] Code đã được format đẹp (indent đúng)

**Kết quả thực nghiệm:**
- [ ] BLEU score ≥ 20% (theo yêu cầu đề bài)
- [ ] Có screenshot từ notebook chứng minh kết quả
- [ ] Biểu đồ loss cho thấy model hội tụ (train/val loss giảm)
- [ ] 5 ví dụ dịch phản ánh đa dạng: tốt, khá, lỗi OOV, lỗi câu dài, lỗi ngữ pháp

**Checkpoint:**
- [ ] `best_model.pth` có thể load được bằng `torch.load()`
- [ ] Checkpoint có kích thước hợp lý (~50-150MB)
- [ ] Đã test load checkpoint và dịch thử 1 câu → kết quả đúng

---

### 📤 CHECKLIST NỘP BÀI:

**Trước khi nộp (48h trước deadline):**
- [ ] Đã chạy lại notebook hoàn chỉnh từ đầu (để chắc chắn không lỗi)
- [ ] Đã export notebook ra PDF (File → Print → Save as PDF)
- [ ] Đã kiểm tra PDF: mở được, không bị lỗi font, hình ảnh hiển thị đúng
- [ ] Đã upload checkpoint lên Google Drive, test link download

**File nộp:**
- [ ] 01 file PDF duy nhất (tên file: MSSV_HoTen_NLP_BaoCao.pdf)
- [ ] Kích thước PDF < 50MB (nếu quá lớn, nén hình ảnh)
- [ ] PDF có bookmark/mục lục (nếu xuất từ Word)

**Nộp trên E-Learning:**
- [ ] Đã login E-Learning, tìm đúng khóa học NLP
- [ ] Đã upload file PDF vào đúng assignment
- [ ] Đã kiểm tra trạng thái: "Submitted for grading"
- [ ] Đã ghi chú: "Link checkpoint: [Google Drive URL]" (nếu có)
- [ ] Nộp TRƯỚC 23:59 ngày 14/12/2025 (đề phòng lỗi hệ thống)

**Sau khi nộp:**
- [ ] Chụp ảnh màn hình submission success
- [ ] Lưu lại bản PDF đã nộp (đề phòng cần resubmit)
- [ ] Giữ nguyên notebook trên Colab (đề phòng giảng viên yêu cầu demo)

---

## ⚠️ LƯU Ý CUỐI CÙNG

### 🚫 TUYỆT ĐỐI TRÁNH:
1. **Nộp trễ** → 0 điểm (không có ngoại lệ)
2. **Sao chép code** → 0 điểm + kỷ luật học tập
3. **Báo cáo thiếu 5 yêu cầu bắt buộc** → mất điểm nặng
4. **Code không chạy được** → mất ≥50% điểm phần implementation

### ✅ ĐỂ ĐẠT ĐIỂM CAO:
1. **Nộp sớm** (1-2 ngày trước deadline) → tránh lỗi hệ thống
2. **Báo cáo chuyên nghiệp**: có đủ 5 yêu cầu bắt buộc, format đẹp, không lỗi chính tả
3. **Code chất lượng**: có comment, dễ đọc, chạy được 100%
4. **Kết quả tốt**: BLEU ≥ 25% (cao hơn yêu cầu 20%)
5. **Phân tích sâu**: 5 ví dụ dịch có phân tích chi tiết, thể hiện hiểu biết về model

---

## 📥 CÁCH XUẤT PDF TỪ NOTEBOOK

### Phương pháp 1: File → Print → Save as PDF

```python
# Trong Colab, chạy cell này để chuẩn bị export
from IPython.display import display, HTML

# Ẩn cell không cần thiết
%%html
<style>
.input {display: none !important;}  /* Ẩn code cells */
</style>

# Sau đó: File → Print → Save as PDF
```

### Phương pháp 2: nbconvert

```bash
# Local machine
jupyter nbconvert --to pdf NLP_Do_An_EnFr_Translation.ipynb

# Hoặc xuất HTML rồi in thành PDF
jupyter nbconvert --to html NLP_Do_An_EnFr_Translation.ipynb
# Mở HTML → Print → Save as PDF
```

### Phương pháp 3: Viết báo cáo riêng bằng Word/LaTeX

**Word:**
- Dễ dàng, WYSIWYG
- Chèn hình, bảng, code dễ dàng
- Nhược điểm: Công thức toán không đẹp

**LaTeX (Overleaf):**
- Công thức toán đẹp
- Format chuyên nghiệp
- Nhược điểm: Học lâu

**Khuyến nghị: Dùng Word + MathType cho công thức**

---

## 🎯 MẪU BÁO CÁO THAM KHẢO

### Link mẫu báo cáo tốt:

1. **Stanford CS224N Project Reports**
   - https://web.stanford.edu/class/cs224n/reports.html
   - Báo cáo sinh viên về NMT, format chuẩn

2. **ACL Anthology (Papers)**
   - https://aclanthology.org/
   - Papers về machine translation, học cách viết academic

3. **Template LaTeX cho NLP**
   - Overleaf: ACL 2023 template
   - https://www.overleaf.com/latex/templates/acl-2023-proceedings-template/

---

## ⏰ TIMELINE ĐỀ XUẤT

**Tổng thời gian: 3-5 ngày**

### Ngày 1: Chuẩn bị (2-3 giờ)
- [ ] Đọc lại requirements
- [ ] Chạy notebook lấy kết quả (BLEU, plots)
- [ ] Screenshot các kết quả quan trọng
- [ ] Thu thập tài liệu tham khảo

### Ngày 2: Viết nháp (4-5 giờ)
- [ ] Phần 1-2: Giới thiệu + Related Work
- [ ] Phần 3: Phương pháp (vẽ sơ đồ)
- [ ] Tạo template Word/LaTeX

### Ngày 3: Viết chính (5-6 giờ)
- [ ] Phần 4: Thực nghiệm (chèn bảng, hình)
- [ ] Phần 5: Kết luận
- [ ] Tài liệu tham khảo
- [ ] Phụ lục

### Ngày 4-5: Hoàn thiện (3-4 giờ)
- [ ] Kiểm tra lỗi chính tả
- [ ] Format lại toàn bộ
- [ ] Đánh số trang, mục lục
- [ ] Export PDF
- [ ] Review lần cuối

---

## 📞 HỖ TRỢ

Nếu cần hỗ trợ thêm:
1. **Vẽ sơ đồ kiến trúc**: Tôi có thể tạo code draw.io hoặc TikZ
2. **Viết công thức LaTeX**: Tôi có thể convert sang LaTeX
3. **Tạo template Word**: Tôi có thể tạo file .docx mẫu
4. **Review báo cáo**: Gửi draft để tôi góp ý

**Chúc bạn viết báo cáo thành công! 🎉**
