# 📦 HƯỚNG DẪN NỘP BÀI - ĐỒ ÁN NLP

**Deadline**: 14/12/2025 (23:59)  
**Hình thức nộp**: E-Learning  
**⚠️ KHÔNG CHẤP NHẬN NỘP TRỄ**

---

## 📋 CHECKLIST TRƯỚC KHI NỘP

### ✅ **Bước 1: Chạy toàn bộ notebook**

1. Mở `NLP_Do_An_EnFr_Translation.ipynb` trên **Google Colab**
2. Chọn Runtime → Change runtime type → **T4 GPU**
3. Upload 6 files data vào `/content/data/` hoặc mount Google Drive
4. Chạy: Runtime → **Run all** (Ctrl+F9)
5. Chờ ~1-2 giờ (training + evaluation)
6. Kiểm tra kết quả:
   - [ ] Training hoàn tất không lỗi
   - [ ] BLEU score hiển thị (VD: 25.3%)
   - [ ] 5 ví dụ dịch hiển thị đầy đủ
   - [ ] Biểu đồ train/val loss hiển thị
   - [ ] Phân tích lỗi hiển thị

---

## 📁 FILES CẦN NỘP

### **1. Notebook (.ipynb)**
**File**: `NLP_Do_An_EnFr_Translation.ipynb`

**Cách download từ Colab:**
- File → Download → Download .ipynb

**Yêu cầu:**
- ✅ Đã chạy hết tất cả cells (có output)
- ✅ Không có cell bị lỗi
- ✅ Có BLEU score kết quả cụ thể
- ✅ Có biểu đồ train/val loss

---

### **2. Báo cáo PDF**
**File**: `NLP_Do_An_EnFr_Translation.pdf`

**Cách tạo từ Colab:**
```
Cách 1 (Khuyến nghị):
- File → Print
- Chọn "Save as PDF"
- Save với tên: NLP_Do_An_EnFr_Translation.pdf

Cách 2:
- File → Download → .ipynb
- Mở bằng Jupyter Notebook local
- File → Download as → PDF via LaTeX
```

**Yêu cầu:**
- ✅ Bao gồm toàn bộ nội dung notebook
- ✅ Code + output + biểu đồ rõ ràng
- ✅ Dưới 50MB (nếu quá lớn, resize hình)

---

### **3. Checkpoint files (.pth)**

**3 files bắt buộc:**
1. `check_point/best_model.pth` (~50-100MB)
2. `check_point/src_vocab.pth` (~200KB)
3. `check_point/tgt_vocab.pth` (~200KB)

**Cách download từ Colab:**
```python
# Chạy cell này để download checkpoints
from google.colab import files
files.download('/content/check_point/best_model.pth')
files.download('/content/check_point/src_vocab.pth')
files.download('/content/check_point/tgt_vocab.pth')
```

**Hoặc:**
- Mở folder `/content/check_point/` bên trái
- Click chuột phải → Download từng file

---

## 📦 CÁCH ĐÓNG GÓI NỘP

### **Cấu trúc folder nộp:**
```
MSSV_HoTen_NLP_Do_An/
│
├── NLP_Do_An_EnFr_Translation.ipynb   ✅ (notebook đã chạy)
├── NLP_Do_An_EnFr_Translation.pdf     ✅ (báo cáo PDF)
│
└── check_point/                        ✅
    ├── best_model.pth
    ├── src_vocab.pth
    └── tgt_vocab.pth
```

### **Đóng gói:**
1. Tạo folder với tên: `MSSV_HoTen_NLP_Do_An`
   - VD: `2033456_NguyenVanA_NLP_Do_An`
2. Copy 4 files vào folder
3. Nén thành file `.zip`:
   - Windows: Click phải → Send to → Compressed folder
   - Mac: Click phải → Compress

### **Kích thước file:**
- Dự kiến: ~100-150MB (với checkpoint)
- Nếu quá 200MB: Kiểm tra lại checkpoint có đúng không

---

## 🚀 CÁCH NỘP LÊN E-LEARNING

1. Đăng nhập E-Learning
2. Vào môn "Xử lý ngôn ngữ tự nhiên"
3. Tìm phần "Nộp đồ án cuối kì"
4. Click "Add submission"
5. Upload file `.zip`
6. Click "Save changes"
7. **Kiểm tra lại:**
   - [ ] File đã upload thành công
   - [ ] Kích thước file hiển thị đúng
   - [ ] Trạng thái: "Submitted for grading"

---

## ⚠️ CÁC LỖI THƯỜNG GẶP VÀ CÁCH KHẮC PHỤC

### **Lỗi 1: Notebook không chạy được**
**Triệu chứng:** Lỗi khi run notebook

**Nguyên nhân:**
- Thiếu file data
- Không chọn GPU
- Library chưa cài đặt

**Cách fix:**
1. Chọn Runtime → Change runtime type → T4 GPU
2. Upload đầy đủ 6 files data
3. Chạy cell cài đặt dependencies trước

---

### **Lỗi 2: Không tạo được PDF**
**Triệu chứng:** File → Print không hoạt động

**Cách fix:**
1. Dùng trình duyệt Chrome/Edge
2. Hoặc: Download .ipynb → Mở bằng Jupyter local → Export PDF
3. Hoặc: Screenshot từng phần → Ghép thành PDF

---

### **Lỗi 3: File quá lớn (>200MB)**
**Triệu chứng:** Upload lên E-Learning bị lỗi

**Cách fix:**
1. Kiểm tra file `best_model.pth` (~50-100MB là bình thường)
2. Xóa file data khỏi folder nộp (KHÔNG NỘP FILE DATA)
3. Nén lại với compression cao hơn

---

### **Lỗi 4: Thiếu checkpoint**
**Triệu chứng:** Training xong nhưng không có file .pth

**Cách fix:**
1. Kiểm tra cell training có chạy hết không
2. Kiểm tra folder `/content/check_point/`
3. Nếu không có: Chạy lại cell training

---

## 📊 TIÊU CHÍ CHẤM ĐIỂM

| Tiêu chí | Điểm | Yêu cầu |
|----------|------|---------|
| Mô hình đúng | 3.0 | Encoder-Decoder LSTM, context vector |
| Data processing | 2.0 | DataLoader, padding/packing, sort batch |
| Training | 1.5 | Early stopping, checkpoint, loss tracking |
| translate() | 1.0 | Greedy decoding, test cases |
| BLEU score | 1.0 | Tính trên test set, có kết quả cụ thể |
| Phân tích lỗi | 1.0 | 5 ví dụ, phân loại, đề xuất cải tiến |
| Code quality | 0.5 | Sạch, comment, cấu trúc rõ |
| Báo cáo | 0.5 | Đầy đủ, rõ ràng, có biểu đồ |
| **TỔNG** | **10.0** | |

---

## 🎯 LỜI KHUYÊN CUỐI CÙNG

### **Nên làm:**
✅ Chạy notebook từ đầu đến cuối ít nhất 1 lần trước khi nộp  
✅ Kiểm tra BLEU score có kết quả hợp lý (≥15%)  
✅ Screenshot kết quả quan trọng (để backup)  
✅ Nộp trước deadline ít nhất 1-2 giờ (phòng lỗi)  
✅ Kiểm tra lại file đã upload thành công chưa

### **Không nên:**
❌ Nộp notebook chưa chạy (không có output)  
❌ Nộp thiếu checkpoint files  
❌ Nộp file bị lỗi (không test trước)  
❌ Nộp trễ (sẽ bị 0 điểm)  
❌ Copy code từ nguồn khác (sẽ bị 0 điểm)

---

## 📞 HỖ TRỢ

**Nếu gặp vấn đề:**
1. Kiểm tra lại hướng dẫn này
2. Đọc file `README.md` trong project
3. Đọc file `COLAB_GUIDE.md` để biết cách chạy trên Colab
4. Hỏi thầy qua email (trước deadline 2 ngày)

---

## ✅ CHECKLIST CUỐI CÙNG TRƯỚC KHI NỘP

- [ ] Đã chạy toàn bộ notebook không lỗi
- [ ] BLEU score hiển thị kết quả cụ thể
- [ ] 5 ví dụ dịch hiển thị rõ ràng
- [ ] Biểu đồ train/val loss hiển thị
- [ ] Có 3 files checkpoint (.pth)
- [ ] Đã export PDF từ notebook
- [ ] Đã đóng gói thành file .zip
- [ ] Tên folder đúng format: MSSV_HoTen_NLP_Do_An
- [ ] Đã upload lên E-Learning thành công
- [ ] Kiểm tra lại trạng thái "Submitted"

---

**Chúc bạn nộp bài thành công và đạt điểm cao! 🎉**

**Deadline**: 14/12/2025 (23:59)  
**Thời gian còn lại**: 7 ngày
