# LaTeX Compilation Guide - NLP Final Project

## 📋 Mục lục
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cách biên dịch](#cách-biên-dịch)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Xử lý lỗi](#xử-lý-lỗi)
- [Sử dụng Overleaf](#sử-dụng-overleaf)

---

## 🖥️ Yêu cầu hệ thống

### LaTeX Distribution
Bạn cần cài đặt một trong các LaTeX distribution sau:

**Windows:**
- [MiKTeX](https://miktex.org/download) (Khuyến nghị) - tự động cài đặt packages khi cần
- [TeX Live](https://www.tug.org/texlive/acquire-netinstall.html) - đầy đủ nhất

**macOS:**
- [MacTeX](https://www.tug.org/mactex/mactex-download.html) - bản TeX Live cho Mac

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install texlive-full texlive-lang-other

# Fedora
sudo dnf install texlive-scheme-full

# Arch Linux
sudo pacman -S texlive-most texlive-lang
```

### Vietnamese Language Support
Đảm bảo có font chữ tiếng Việt:
- **Windows**: Times New Roman đã có sẵn
- **macOS/Linux**: Cài đặt Microsoft Core Fonts:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ttf-mscorefonts-installer
  
  # macOS (với Homebrew)
  brew tap homebrew/cask-fonts
  brew install font-times-new-roman
  ```

---

## 📦 Cài đặt

### Option 1: MiKTeX (Windows - Khuyến nghị)

1. **Download và cài đặt MiKTeX:**
   - Tải từ https://miktex.org/download
   - Chạy installer, chọn "Install missing packages on-the-fly: Yes"

2. **Cài đặt TeXworks hoặc VS Code:**
   - MiKTeX đi kèm TeXworks (đơn giản)
   - Hoặc dùng VS Code với extension **LaTeX Workshop**

### Option 2: TeX Live (Cross-platform)

1. **Download TeX Live:**
   - Windows/Linux: https://www.tug.org/texlive/acquire-netinstall.html
   - macOS: Dùng MacTeX thay thế

2. **Cài đặt (mất ~4GB):**
   ```bash
   # Linux
   sudo ./install-tl
   
   # Chọn scheme: full
   ```

### Option 3: Overleaf (Online - Không cần cài đặt)

Xem phần [Sử dụng Overleaf](#sử-dụng-overleaf) bên dưới.

---

## 🔨 Cách biên dịch

### Biên dịch trên Command Line

**Cách 1: Biên dịch với XeLaTeX (Khuyến nghị cho tiếng Việt)**
```bash
# Di chuyển vào thư mục latex
cd latex

# Biên dịch 4 lần để references đầy đủ
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

**Cách 2: Biên dịch với pdfLaTeX**
```bash
cd latex

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Giải thích các bước:**
1. `xelatex main.tex` - Biên dịch lần 1 (tạo .aux, .toc)
2. `bibtex main` - Xử lý bibliography
3. `xelatex main.tex` - Biên dịch lần 2 (thêm citations)
4. `xelatex main.tex` - Biên dịch lần 3 (hoàn thiện references, page numbers)

**Output:** File `main.pdf` sẽ được tạo trong thư mục `latex/`

### Biên dịch với VS Code

1. **Cài đặt extension:**
   - Mở VS Code
   - Search "LaTeX Workshop" trong Extensions
   - Install

2. **Cấu hình (tạo `.vscode/settings.json`):**
   ```json
   {
     "latex-workshop.latex.recipes": [
       {
         "name": "xelatex -> bibtex -> xelatex*2",
         "tools": [
           "xelatex",
           "bibtex",
           "xelatex",
           "xelatex"
         ]
       }
     ],
     "latex-workshop.latex.tools": [
       {
         "name": "xelatex",
         "command": "xelatex",
         "args": [
           "-synctex=1",
           "-interaction=nonstopmode",
           "-file-line-error",
           "%DOC%"
         ]
       },
       {
         "name": "bibtex",
         "command": "bibtex",
         "args": ["%DOCFILE%"]
       }
     ]
   }
   ```

3. **Biên dịch:**
   - Mở file `main.tex`
   - Nhấn `Ctrl+Alt+B` (hoặc `Cmd+Option+B` trên Mac)
   - Hoặc click nút ▶️ "Build LaTeX project" trên toolbar
   - Preview: `Ctrl+Alt+V`

### Biên dịch với TeXworks (MiKTeX)

1. Mở `main.tex` trong TeXworks
2. Chọn compiler: **XeLaTeX** trong dropdown (góc trên bên trái)
3. Nhấn nút ▶️ (hoặc `Ctrl+T`)
4. Sau lần compile đầu, chọn **BibTeX** rồi compile
5. Chuyển lại **XeLaTeX** và compile 2 lần nữa

---

## 📁 Cấu trúc dự án

```
latex/
├── main.tex                    # File chính (bắt đầu từ đây)
│
├── chapters/                   # 5 chương chính
│   ├── 1_introduction.tex      # Chương 1: Giới thiệu
│   ├── 2_related_work.tex      # Chương 2: Các công trình liên quan
│   ├── 3_methodology.tex       # Chương 3: Phương pháp tiếp cận
│   ├── 4_experiments.tex       # Chương 4: Thực nghiệm và kết quả
│   └── 5_conclusion.tex        # Chương 5: Kết luận
│
├── appendices/                 # 3 phụ lục
│   ├── appendix_a.tex          # Phụ lục A: Cấu hình
│   ├── appendix_b.tex          # Phụ lục B: Mã nguồn
│   └── appendix_c.tex          # Phụ lục C: Checkpoint links
│
├── references.bib              # Bibliography (15 papers)
│
└── figures/                    # (Optional) Thư mục cho hình ảnh
    ├── architecture.png        # Nếu có hình vẽ riêng
    └── loss_plot.png           # Thay vì dùng TikZ
```

---

## 🐛 Xử lý lỗi

### Lỗi 1: `! LaTeX Error: File 'babel.sty' not found`

**Nguyên nhân:** Thiếu package

**Giải pháp:**
```bash
# MiKTeX
mpm --install=babel

# TeX Live
tlmgr install babel babel-vietnamese
```

### Lỗi 2: `Package babel Error: Unknown option 'vietnamese'`

**Nguyên nhân:** Thiếu language package

**Giải pháp:**
```bash
# MiKTeX
mpm --install=babel-vietnamese

# TeX Live
tlmgr install babel-vietnamese
```

### Lỗi 3: Font không hiển thị tiếng Việt

**Giải pháp:**
1. Thay đổi compiler từ `pdflatex` sang `xelatex`
2. Hoặc cài đặt `vntex`:
   ```bash
   # TeX Live
   tlmgr install vntex
   ```

### Lỗi 4: `Undefined control sequence \cite`

**Nguyên nhân:** Chưa chạy `bibtex`

**Giải pháp:**
```bash
xelatex main.tex
bibtex main      # <-- Bước này quan trọng
xelatex main.tex
xelatex main.tex
```

### Lỗi 5: Missing TikZ/pgfplots

**Giải pháp:**
```bash
# MiKTeX
mpm --install=pgf pgfplots

# TeX Live
tlmgr install pgf pgfplots
```

### Lỗi 6: Compile quá lâu

**Nguyên nhân:** TikZ diagrams phức tạp

**Giải pháp:**
- Comment các hình TikZ khi đang viết text
- Hoặc dùng `\includeonly{chapters/1_introduction}` để compile 1 chương

---

## ☁️ Sử dụng Overleaf

### Cách 1: Upload Project

1. Đăng ký tài khoản tại https://www.overleaf.com/
2. Click **"New Project"** → **"Upload Project"**
3. Zip toàn bộ thư mục `latex/`:
   ```bash
   cd latex
   zip -r nlp_project.zip .
   ```
4. Upload file `nlp_project.zip`
5. Chọn compiler: **Menu** (góc trên bên trái) → **Compiler** → **XeLaTeX**
6. Click **"Recompile"**

### Cách 2: Tạo project mới

1. **New Project** → **Blank Project**
2. Tạo cấu trúc thư mục:
   - Upload `main.tex`
   - Tạo folder `chapters/` và upload 5 file
   - Tạo folder `appendices/` và upload 3 file
   - Upload `references.bib`
3. Chọn compiler: **XeLaTeX**
4. Compile

### Lợi ích của Overleaf:
- ✅ Không cần cài đặt
- ✅ Tự động cài packages
- ✅ Real-time preview
- ✅ Collaborative editing
- ✅ Version history

---

## 📊 Kích thước PDF dự kiến

- **Pages**: ~35-40 trang
- **File size**: ~2-3 MB (với TikZ diagrams)
- **Compile time**: 
  - MiKTeX/TeX Live: ~30-60 giây (lần đầu)
  - Overleaf: ~10-15 giây

---

## 📝 Chỉnh sửa nội dung

### Thay đổi thông tin sinh viên

Mở `main.tex`, tìm dòng:
```latex
\Large \textbf{Họ và tên SV 1}\\
Mã số SV: 123456789\\
\Large \textbf{Họ và tên SV 2}\\
Mã số SV: 987654321\\
```

Sửa thành thông tin của bạn.

### Thêm hình ảnh

1. **Tạo thư mục figures/**:
   ```bash
   mkdir -p latex/figures
   ```

2. **Thêm hình vào LaTeX**:
   ```latex
   \begin{figure}[H]
   \centering
   \includegraphics[width=0.8\textwidth]{figures/loss_plot.png}
   \caption{Training and Validation Loss}
   \label{fig:loss}
   \end{figure}
   ```

### Thêm citation mới

1. **Thêm vào `references.bib`**:
   ```bibtex
   @article{your2024paper,
     title={Your Paper Title},
     author={Author, Name},
     journal={Journal Name},
     year={2024}
   }
   ```

2. **Cite trong text**:
   ```latex
   According to \cite{your2024paper}, ...
   ```

---

## 🎯 Checklist trước khi nộp

- [ ] Compile thành công không có errors
- [ ] Tất cả citations hiển thị đúng (không có `[?]`)
- [ ] Tất cả figures/tables có captions
- [ ] Thông tin sinh viên đã cập nhật
- [ ] Google Drive links đã thêm (appendix_c.tex)
- [ ] File PDF < 10 MB
- [ ] Tất cả 5 mandatory components có (architecture, loss plot, BLEU, 5 examples, source code)

---

## 💡 Tips

1. **Compile nhanh hơn**: Dùng `\includeonly` khi đang viết:
   ```latex
   \includeonly{chapters/1_introduction}
   ```

2. **Xem lỗi chi tiết**: Check file `main.log` nếu compile fail

3. **Backup thường xuyên**: Commit vào Git sau mỗi chương:
   ```bash
   git add .
   git commit -m "Completed Chapter 1"
   ```

4. **Preview sections**: Dùng `\input` thay vì `\include` để compile nhanh

---

## 📧 Hỗ trợ

Nếu gặp vấn đề:
1. Check file `main.log` để xem error cụ thể
2. Google error message + "latex"
3. Hỏi trên [TeX StackExchange](https://tex.stackexchange.com/)
4. Hoặc email: your.email@example.com

---

**Good luck! 🚀**

Hạn nộp: **14/12/2025 - 23:59**
