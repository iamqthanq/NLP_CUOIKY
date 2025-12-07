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
├── data/                       # Dữ liệu huấn luyện
│   ├── train.en / train.fr
│   ├── val.en / val.fr
│   └── test.en / test.fr
│
├── src/                        # Source code
│   ├── config.py              # Cấu hình (Task 1) ✅
│   ├── utils.py               # Utility functions (Task 1) ✅
│   ├── data_loader.py         # Data processing (Task 2) ✅
│   ├── model.py               # Encoder-Decoder LSTM (Task 3)
│   ├── train.py               # Training loop (Task 4)
│   └── evaluate.py            # Evaluation & translate() (Task 5)
│
├── check_point/               # Lưu model checkpoints
│   ├── src_vocab.pth
│   ├── tgt_vocab.pth
│   └── best_model.pth
│
├── report/                    # Báo cáo PDF
│
└── requirements.txt           # Dependencies ✅
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

**Trên máy local (Windows):**
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

**Trên Google Colab:**
```python
# Clone/upload project lên Colab
# Cài đặt dependencies
!pip install spacy torch torchtext
!python -m spacy download en_core_web_sm
!python -m spacy download fr_core_news_sm

# Upload data files vào /content/data/
```

### 2. Kiểm tra data loading (Task 1 & 2) ✅

```powershell
cd src
python data_loader.py
```

Kết quả mong đợi:
- Build vocabulary: English ~10,000 tokens, French ~10,000 tokens
- DataLoader: batch size 64, sorted by length
- Test một batch thành công

### 3. Training model (Task 3 & 4)

```powershell
python train.py
```

### 4. Evaluation & Translation (Task 5)

```powershell
python evaluate.py
```

## 📊 Tiến độ hiện tại

- [x] **Task 1**: Thiết lập môi trường + cấu trúc project
  - ✅ `config.py`: Cấu hình đầy đủ theo yêu cầu
  - ✅ `utils.py`: Vocabulary, tokenization, helper functions
  - ✅ `requirements.txt`: Dependencies

- [x] **Task 2**: Xử lý dữ liệu & DataLoader
  - ✅ Tokenization đơn giản (lowercase + split)
  - ✅ Vocabulary building (giới hạn 10,000 từ phổ biến nhất)
  - ✅ Special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>`
  - ✅ Padding/Packing: Sort batch theo độ dài, sử dụng pack_padded_sequence
  - ✅ DataLoader: batch size 32-128 (mặc định 64)

- [ ] **Task 3**: Xây dựng mô hình Encoder-Decoder LSTM
- [ ] **Task 4**: Viết vòng train + val, early stopping, checkpoint
- [ ] **Task 5**: Viết hàm translate() + greedy decoding
- [ ] **Task 6**: Đánh giá BLEU score + biểu đồ loss
- [ ] **Task 7**: Phân tích 5 ví dụ lỗi + đề xuất cải tiến
- [ ] **Task 8**: Lưu checkpoint + export báo cáo PDF

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

## 📈 Thang điểm (10 điểm)

1. **Triển khai mô hình Encoder-Decoder LSTM** (3.0đ)
2. **Xử lý dữ liệu, DataLoader, padding/packing** (2.0đ) ✅
3. **Huấn luyện ổn định, early stopping, checkpoint** (1.5đ)
4. **Hàm translate() hoạt động với câu mới** (1.0đ)
5. **Đánh giá BLEU score + biểu đồ loss** (1.0đ)
6. **Phân tích 5 ví dụ lỗi + đề xuất cải tiến** (1.0đ)
7. **Chất lượng mã nguồn (sạch, comment, cấu trúc)** (0.5đ) ✅
8. **Báo cáo (đầy đủ, rõ ràng, biểu đồ, trích dẫn)** (0.5đ)
9. **Điểm cộng (mở rộng: attention/beam search)** (1.0đ)

## 📝 Lưu ý quan trọng

1. ⚠️ **VẤN ĐỀ DATA**: File `val.fr` bị thiếu → Cần tải lại từ dataset gốc
2. ✅ Mã nguồn phải chạy được từ đầu đến cuối trên Google Colab hoặc máy local
3. ✅ Báo cáo PDF phải bao gồm: sơ đồ kiến trúc, biểu đồ loss, BLEU score, 5 ví dụ dịch, phân tích lỗi
4. ✅ Checkpoint mô hình (`best_model.pth`) bắt buộc nộp
5. ❌ Không sao chép mã → Sẽ bị 0 điểm

## 📚 Tài liệu tham khảo

- Sutskever et al. (2014). *Sequence to Sequence Learning with Neural Networks*
- PyTorch Documentation: [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- Multi30K Dataset: [https://github.com/multi30k/dataset](https://github.com/multi30k/dataset)

---

**Deadline**: 14/12/2025 (23:59)  
**Hình thức nộp**: 01 file PDF + mã nguồn (zip) qua E-Learning  
**Không chấp nhận nộp trễ**
