# Đồ án NLP - Dịch máy Anh-Pháp với LSTM Encoder-Decoder

## 📋 Giới thiệu
Đồ án xử lý ngôn ngữ tự nhiên: Xây dựng mô hình Encoder-Decoder LSTM với context vector cố định để dịch từ tiếng Anh sang tiếng Pháp.

**Dataset**: Multi30K (en-fr)
- Train: 29,000 cặp câu
- Validation: 1,000 cặp câu  
- Test: 1,000 cặp câu

## 🎯 Mục tiêu
1. Hiểu và triển khai Encoder-Decoder LSTM với context vector cố định
2. Xử lý dữ liệu chuỗi, huấn luyện, đánh giá bằng BLEU score
3. Phân tích lỗi dịch thuật và đề xuất cải tiến (attention, beam search...)

## 📁 Cấu trúc project

```
NLP_DO_AN/
│
├── data/                       # Dữ liệu huấn luyện (31,014 câu)
│   ├── train.en / train.fr    # 29,000 cặp câu
│   ├── val.en / val.fr        # 1,014 cặp câu
│   └── test.en / test.fr      # 1,000 cặp câu
│
├── src/                        # Source code (legacy - đã tích hợp vào notebook)
│   ├── config.py              # Cấu hình ✅
│   ├── utils.py               # Utility functions ✅
│   └── data_loader.py         # Data processing ✅
│
├── check_point/               # Lưu model checkpoints
│   ├── src_vocab.pth          # Từ điển tiếng Anh (10,000 tokens)
│   ├── tgt_vocab.pth          # Từ điển tiếng Pháp (10,000 tokens)
│   └── best_model.pth         # Model weights tốt nhất
│
├── report/                    # Báo cáo & tài liệu
│   ├── PROJECT_OVERVIEW.md    # Tổng quan dự án
│   ├── CODE_EXPLANATION.md    # Giải thích code chi tiết
│   ├── COLAB_GUIDE.md         # Hướng dẫn chạy trên Colab
│   └── PROGRESS_REPORT.md     # Báo cáo tiến độ
│
├── NLP_Do_An_EnFr_Translation.ipynb  # ⭐ NOTEBOOK CHÍNH (2,045 dòng) ✅
│                                      # Chứa TOÀN BỘ 8 tasks hoàn chỉnh
│
├── requirements.txt           # Dependencies ✅
├── README.md                  # File này
├── COLAB_GUIDE.md            # Hướng dẫn Colab
├── PROJECT_OVERVIEW.md       # Tổng quan
└── CODE_EXPLANATION.md       # Giải thích code
```

### 📓 **File quan trọng nhất:**
**`NLP_Do_An_EnFr_Translation.ipynb`** - Notebook Jupyter hoàn chỉnh, sẵn sàng chạy trên Google Colab hoặc local

## 🚀 Hướng dẫn sử dụng

### ⭐ **KHUYẾN NGHỊ: Sử dụng Google Colab**

**File notebook đã hoàn chỉnh:** `NLP_Do_An_EnFr_Translation.ipynb`

#### **Bước 1: Mở notebook trên Google Colab**
1. Truy cập [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Chọn `NLP_Do_An_EnFr_Translation.ipynb`

#### **Bước 2: Chọn GPU Runtime**
1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU**
3. Save

#### **Bước 3: Upload dữ liệu**
Có 2 cách:

**Cách 1: Upload trực tiếp (nhanh, dùng cho demo)**
- Cell đầu tiên có hướng dẫn upload 6 files data
- Drag & drop vào folder `/content/data/`

**Cách 2: Sử dụng Google Drive (khuyến nghị)**
- Mount Google Drive
- Tạo folder `MyDrive/NLP_Do_An/data/`
- Upload 6 files data vào đó
- Notebook sẽ tự động link

#### **Bước 4: Chạy toàn bộ notebook**
1. Runtime → Run all (Ctrl+F9)
2. Chờ ~1-2 giờ (training với GPU T4)
3. Xem kết quả:
   - BLEU score
   - 5 ví dụ dịch
   - Phân tích lỗi
   - Đề xuất cải tiến

#### **Bước 5: Export kết quả**
1. File → Download → `.ipynb` (notebook)
2. File → Print → Save as PDF (báo cáo)
3. Download checkpoint từ `/content/check_point/best_model.pth`

---

### 💻 **Chạy trên máy Local (Optional)**

**Yêu cầu:**
- Python 3.8+
- GPU NVIDIA với CUDA (khuyến nghị) hoặc chấp nhận training chậm
- RAM >= 8GB

**Cài đặt:**
```powershell
# Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

**Chạy notebook:**
```powershell
# Khởi động Jupyter
jupyter notebook

# Mở file NLP_Do_An_EnFr_Translation.ipynb
# Run All Cells
```

**Lưu ý:** Training trên CPU sẽ mất ~10-20 giờ thay vì 1-2 giờ với GPU

## 📊 Tiến độ hiện tại

### ✅ **HOÀN THÀNH 100% YÊU CẦU CƠ BẢN (10/10 ĐIỂM)**

- [x] **Task 1**: Triển khai mô hình Encoder-Decoder LSTM (3.0đ)
  - ✅ `Encoder` class: LSTM 2 layers, embedding 256, hidden 512
  - ✅ `Decoder` class: LSTM 2 layers, Linear output layer
  - ✅ `Seq2Seq` class: Context vector từ Encoder → Decoder
  - ✅ Teacher forcing ratio: 0.5
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 3

- [x] **Task 2**: Xử lý dữ liệu & DataLoader (2.0đ)
  - ✅ Tokenization đơn giản (lowercase + regex)
  - ✅ Vocabulary building (giới hạn 10,000 từ phổ biến nhất)
  - ✅ Special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>`
  - ✅ Padding/Packing: Sort batch theo độ dài giảm dần
  - ✅ DataLoader: batch size 64, sử dụng pack_padded_sequence
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 2

- [x] **Task 3**: Huấn luyện ổn định, early stopping, checkpoint (1.5đ)
  - ✅ Loss: CrossEntropyLoss(ignore_index=pad_idx)
  - ✅ Optimizer: Adam(lr=0.001)
  - ✅ Early stopping: patience=3 epochs
  - ✅ Save best model: `best_model.pth`
  - ✅ Tracking: Train/val loss + Perplexity
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 4

- [x] **Task 4**: Hàm translate() hoạt động với câu mới (1.0đ)
  - ✅ Greedy decoding: Chọn token xác suất cao nhất
  - ✅ Dừng khi gặp `<eos>` hoặc max_len=50
  - ✅ Test với 3 câu mẫu cụ thể
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 5

- [x] **Task 5**: Đánh giá BLEU score + biểu đồ loss (1.0đ)
  - ✅ BLEU score: Sử dụng `nltk.translate.bleu_score`
  - ✅ Tính trên test set (200+ câu)
  - ✅ Hiển thị 5 ví dụ dịch với BLEU từng câu
  - ✅ Biểu đồ matplotlib: Train/val loss
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 6

- [x] **Task 6**: Phân tích lỗi + đề xuất cải tiến (1.0đ)
  - ✅ Phân loại 4 loại lỗi: OOV, Câu dài, Ngữ pháp, Dịch tốt
  - ✅ Hiển thị ví dụ cụ thể cho mỗi loại
  - ✅ Đề xuất 5 cải tiến chi tiết:
    1. Attention mechanism (Luong/Bahdanau)
    2. Subword tokenization (BPE)
    3. Beam search (beam_size=3-5)
    4. Tăng dữ liệu (WMT 2014)
    5. Scheduled sampling
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 7

- [x] **Task 7**: Chất lượng mã nguồn (0.5đ)
  - ✅ Cấu trúc rõ ràng: 8 bước từ setup → tổng hợp
  - ✅ Comment chi tiết (tiếng Việt + tiếng Anh)
  - ✅ Naming conventions chuẩn Python
  - ✅ Docstring đầy đủ cho mọi function
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 7.5

- [x] **Task 8**: Báo cáo tổng hợp (0.5đ)
  - ✅ Tổng hợp toàn bộ kết quả
  - ✅ Thống kê: Model architecture, training config, performance
  - ✅ Hướng dẫn sử dụng
  - ✅ File: `NLP_Do_An_EnFr_Translation.ipynb` - BƯỚC 8

### 📓 **FILE NOTEBOOK CHÍNH**

**`NLP_Do_An_EnFr_Translation.ipynb`** (2,045 dòng, 8 bước hoàn chỉnh)

Cấu trúc notebook:
```
BƯỚC 1: Thao tác ban đầu (GPU, Drive, Data upload)
BƯỚC 2: Cài đặt Dependencies + Config + Utils + DataLoader
BƯỚC 3: Xây dựng mô hình (Encoder, Decoder, Seq2Seq)
BƯỚC 4: Huấn luyện (Training loop với Early Stopping)
BƯỚC 5: Dịch câu mới (translate() + test 3 câu)
BƯỚC 6: Đánh giá BLEU score (calculate_bleu + 5 ví dụ)
BƯỚC 7: Phân tích lỗi + Đề xuất cải tiến
BƯỚC 7.5: Đánh giá chất lượng code
BƯỚC 8: Tổng hợp kết quả
```

### ❌ **PHẦN MỞ RỘNG (TÙY CHỌN +1 ĐIỂM) - CHƯA LÀM**

- [ ] Dataset WMT 2014 (4.5M câu thay vì Multi30K 29K)
- [ ] Tăng số layer LSTM (4-6 layers) hoặc hidden size (1024)
- [ ] Beam search thay greedy decoding
- [ ] Attention mechanism (Luong/Bahdanau)
- [ ] So sánh performance với/không có attention

**Lưu ý:** Phần mở rộng KHÔNG BẮT BUỘC, chỉ làm nếu muốn điểm tối đa 11/10

## 🔧 Cấu hình mô hình

**Theo yêu cầu đề bài:**
- Embedding dimension: 256-512 (mặc định 256)
- Hidden size: 512
- Number of LSTM layers: 2
- Dropout: 0.3-0.5 (mặc định 0.3)
- Teacher forcing ratio: 0.5
- Optimizer: Adam(lr=0.001)
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=3 epochs
- Loss: CrossEntropyLoss (ignore_index=pad_idx)

**Training:**
- Epochs: 10-20 (mặc định 15)
- Batch size: 32-128 (mặc định 64)
- Max sequence length: 50
- Vocab size: 10,000 (mỗi ngôn ngữ)

## 💡 Khuyến nghị môi trường

### ✅ **Google Colab** (Khuyến nghị)
**Ưu điểm:**
- GPU miễn phí (T4/P100) → Training nhanh hơn 10-20 lần
- Không cần setup môi trường phức tạp
- Dễ export notebook (.ipynb) + PDF báo cáo
- Checkpoint lưu trực tiếp Google Drive

**Cách chuyển sang Colab:**
1. Tạo notebook mới trên Colab
2. Upload thư mục `data/` lên `/content/data/`
3. Copy code từ `src/*.py` vào các cells
4. Run từ đầu đến cuối
5. Export notebook + PDF

### 🖥️ **Máy Local**
**Ưu điểm:**
- Code/debug nhanh
- Toàn quyền kiểm soát

**Yêu cầu:**
- GPU NVIDIA (khuyến nghị) hoặc chấp nhận training chậm
- Python >= 3.8
- PyTorch với CUDA support

## 📈 Thang điểm đánh giá

### ✅ **ĐIỂM CƠ BẢN (10/10 - ĐÃ HOÀN THÀNH)**

| # | Tiêu chí | Điểm | Trạng thái | Vị trí trong Notebook |
|---|----------|------|------------|----------------------|
| 1 | Triển khai mô hình Encoder-Decoder LSTM | 3.0đ | ✅ | BƯỚC 3 |
| 2 | Xử lý dữ liệu, DataLoader, padding/packing | 2.0đ | ✅ | BƯỚC 2 |
| 3 | Huấn luyện ổn định, early stopping, checkpoint | 1.5đ | ✅ | BƯỚC 4 |
| 4 | Hàm translate() hoạt động với câu mới | 1.0đ | ✅ | BƯỚC 5 |
| 5 | Đánh giá BLEU score + biểu đồ loss | 1.0đ | ✅ | BƯỚC 6 |
| 6 | Phân tích 5 ví dụ lỗi + đề xuất cải tiến | 1.0đ | ✅ | BƯỚC 7 |
| 7 | Chất lượng mã nguồn (sạch, comment, cấu trúc) | 0.5đ | ✅ | BƯỚC 7.5 |
| 8 | Báo cáo (đầy đủ, rõ ràng, biểu đồ, trích dẫn) | 0.5đ | ✅ | BƯỚC 8 |
| | **TỔNG** | **10.0đ** | **✅ HOÀN THÀNH** | |

### ⭐ **ĐIỂM MỞ RỘNG (TÙY CHỌN +1 ĐIỂM - CHƯA LÀM)**

| # | Nội dung mở rộng | Điểm | Trạng thái |
|---|------------------|------|------------|
| 1 | Dataset WMT 2014 (4.5M câu) | +0.3đ | ❌ |
| 2 | Tăng số layer LSTM hoặc hidden size | +0.2đ | ❌ |
| 3 | Beam search (beam_size=3-5) | +0.2đ | ❌ |
| 4 | Attention mechanism (Luong/Bahdanau) | +0.2đ | ❌ |
| 5 | So sánh performance với/không attention | +0.1đ | ❌ |
| | **TỔNG MỞ RỘNG** | **+1.0đ** | **Không bắt buộc** |

**Lưu ý:** Phần mở rộng CHỈ LÀM nếu muốn điểm tối đa 11/10. Đã có đề xuất chi tiết trong BƯỚC 7 của notebook.

## 📝 Lưu ý quan trọng

### ✅ **ĐÃ HOÀN THÀNH:**
1. ✅ Mã nguồn notebook chạy được từ đầu đến cuối trên Google Colab với GPU T4
2. ✅ Notebook chứa đầy đủ: sơ đồ kiến trúc (comment), biểu đồ loss, BLEU score, 5 ví dụ dịch, phân tích lỗi
3. ✅ Checkpoint mô hình sẽ được save tự động: `check_point/best_model.pth`
4. ✅ Code tự viết, có comment chi tiết (tiếng Việt + tiếng Anh)

### ⚠️ **CẦN LƯU Ý KHI NỘP:**
1. 📄 **File nộp bắt buộc:**
   - `NLP_Do_An_EnFr_Translation.ipynb` (notebook)
   - `NLP_Do_An_EnFr_Translation.pdf` (export từ notebook)
   - `check_point/best_model.pth` (model weights)
   - `check_point/src_vocab.pth` (English vocabulary)
   - `check_point/tgt_vocab.pth` (French vocabulary)

2. ⏱️ **Deadline:** 14/12/2025 (23:59) - KHÔNG CHẤP NHẬN NỘP TRỄ

3. 📊 **Cách export PDF từ Colab:**
   - File → Print
   - Chọn "Save as PDF"
   - Hoặc: File → Download → .ipynb rồi mở bằng Jupyter Notebook → Export as PDF

4. 🎯 **Kiểm tra trước khi nộp:**
   - [ ] Notebook chạy được từ đầu đến cuối (Runtime → Run all)
   - [ ] Có BLEU score kết quả cụ thể (VD: 25.3%)
   - [ ] Có 5 ví dụ dịch hiển thị rõ ràng
   - [ ] Có biểu đồ train/val loss
   - [ ] Có checkpoint files (3 files .pth)

5. ❌ **Tránh sai sót:**
   - Không nộp thiếu file
   - Không nộp file bị lỗi (không chạy được)
   - Không sao chép code từ nguồn khác → 0 điểm

## 📚 Tài liệu tham khảo

- Sutskever et al. (2014). *Sequence to Sequence Learning with Neural Networks*
- PyTorch Documentation: [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- Multi30K Dataset: [https://github.com/multi30k/dataset](https://github.com/multi30k/dataset)

---

**Deadline**: 14/12/2025 (23:59)  
**Hình thức nộp**: 01 file PDF + mã nguồn (zip) qua E-Learning  
**Không chấp nhận nộp trễ**
