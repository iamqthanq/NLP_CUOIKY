# 📘 Hướng Dẫn Biên Dịch Báo Cáo LaTeX

## ✅ File đã được cập nhật với kết quả thực tế

### Các thông số chính đã update:
- **Vocabulary size**: 15,000 (thay vì 10,000)
- **Model architecture**: 
  - Embedding: 512 dim
  - Hidden: 1024 dim  
  - Layers: 3 (thay vì 2)
  - Dropout: 0.5
- **Training**:
  - Epochs: 20 (early stop at epoch 17)
  - Batch size: 128
  - Teacher forcing: 0.7 → 0.5 (scheduled sampling)
  - Early stopping patience: 5
- **BLEU Scores**:
  - Vanilla (No Attention): 29.12%
  - With Attention: 36.57%
  - Improvement: +7.45% (25.6% relative)

## 🚀 Cách 1: Upload lên Overleaf (Khuyến nghị)

1. Truy cập https://www.overleaf.com
2. Tạo project mới → Upload Project
3. Upload file `latex_report.zip`
4. Overleaf sẽ tự động giải nén
5. Click nút "Recompile" để tạo PDF

**Compiler**: XeLaTeX hoặc pdfLaTeX (cả 2 đều ok)

## 🖥️ Cách 2: Compile trên máy local

### Windows (MiKTeX):
```powershell
cd latex
xelatex main.tex
bibtex main
xelatex main.tex  
xelatex main.tex
```

### macOS/Linux (TeX Live):
```bash
cd latex
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

**Output**: `main.pdf` sẽ được tạo trong thư mục `latex/`

## 📁 Cấu trúc thư mục

```
latex/
├── main.tex                 # File chính
├── references.bib           # Tài liệu tham khảo
├── chapters/                # 5 chương
│   ├── 1_introduction.tex
│   ├── 2_related_work.tex
│   ├── 3_methodology.tex    # ✅ Updated: vocab 15K, 3 layers, 1024 hidden
│   ├── 4_experiments.tex    # ✅ Updated: BLEU 29.12% vs 36.57%
│   └── 5_conclusion.tex     # ✅ Updated: kết quả đạt 11/10 điểm
├── appendices/              # 3 phụ lục
│   ├── appendix_a.tex       # ✅ Updated: config thực tế
│   ├── appendix_b.tex       # Code examples
│   └── appendix_c.tex       # Checkpoints
└── README.md                # Hướng dẫn chi tiết
```

## ✅ Các thay đổi quan trọng đã thực hiện

### Chapter 3 (Methodology):
- ✅ Vocab size: 10K → 15K
- ✅ Layers: 2 → 3
- ✅ Hidden dim: 512 → 1024
- ✅ Embedding: 256 → 512
- ✅ Dropout: 0.3 → 0.5
- ✅ Teacher forcing: 0.5 → 0.7 (scheduled sampling)
- ✅ Early stopping patience: 3 → 5

### Chapter 4 (Experiments):
- ✅ Training loss curve: Updated với 17 epochs
- ✅ BLEU Vanilla: 29.12%
- ✅ BLEU Attention: 36.57%
- ✅ So sánh theo độ dài câu:
  - Medium (6-10 từ): 38.79% → 44.57% (+5.78%)
  - Long (>10 từ): 28.46% → 35.98% (+7.52%)
- ✅ 5 ví dụ dịch thực tế

### Chapter 5 (Conclusion):
- ✅ Kết quả: 11/10 điểm (10 cơ bản + 1 mở rộng)
- ✅ Attention đã implement → không còn trong future work
- ✅ Beam search đã implement (K=5)

### Appendix A (Configuration):
- ✅ Tất cả thông số updated theo code thực tế
- ✅ Tổng tham số: ~61.7M (thay vì 20M)

## 🐛 Xử lý lỗi compile

### Lỗi "File not found":
- Đảm bảo tất cả files trong cấu trúc thư mục đúng
- Check đường dẫn relative paths

### Lỗi tiếng Việt:
- Dùng XeLaTeX thay vì pdfLaTeX
- Hoặc uncomment dòng `\usepackage[vietnamese]{babel}` trong main.tex

### Lỗi references:
- Chạy `bibtex main` sau lần compile đầu
- Compile lại 2 lần nữa để references được update

## 📧 Liên hệ

Nếu có vấn đề khi compile, kiểm tra:
1. File `main.pdf` đã được tạo chưa?
2. Check log file: `main.log`
3. Nếu dùng Overleaf, check "Logs and output files"

---

**Cập nhật**: December 9, 2025
**Version**: Final (với kết quả thực tế từ code)
