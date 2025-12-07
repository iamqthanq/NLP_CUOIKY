# 📖 GIẢI THÍCH CHI TIẾT TOÀN BỘ CODE

## 📁 MỤC LỤC

1. [config.py - Cấu hình toàn bộ project](#1-configpy)
2. [utils.py - Các hàm tiện ích](#2-utilspy)
3. [data_loader.py - Xử lý dữ liệu và DataLoader](#3-data_loaderpy)
4. [Luồng hoạt động tổng thể](#4-luồng-hoạt-động-tổng-thể)
5. [Câu hỏi thường gặp](#5-câu-hỏi-thường-gặp)

---

## 1. config.py

### 🎯 Mục đích
File cấu hình chứa **TẤT CẢ** các tham số của project. Khi muốn thay đổi batch size, learning rate, số epochs... chỉ cần sửa file này.

### 📝 Chi tiết từng phần

#### **1.1. Import và Path Configuration**

```python
import torch
from pathlib import Path
import os

# Tương thích cả local và Colab
try:
    # Nếu chạy từ file .py (local)
    BASE_DIR = Path(__file__).parent.parent
except NameError:
    # Nếu chạy trên Colab/Jupyter (không có __file__)
    BASE_DIR = Path("/content")
```

**Giải thích:**
- `__file__`: Biến Python chứa đường dẫn file hiện tại
- **Local**: `BASE_DIR = d:\Corel\HK1_NAM3\NLP\NLP_DO_AN\`
- **Colab**: `BASE_DIR = /content/`
- `try-except`: Xử lý lỗi khi `__file__` không tồn tại (Colab/Jupyter)

**Tại sao cần?**
- Code chạy được cả local và Colab mà không cần sửa đường dẫn

---

#### **1.2. Data Files**

```python
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "check_point"

TRAIN_EN = DATA_DIR / "train.en"
TRAIN_FR = DATA_DIR / "train.fr"
VAL_EN = DATA_DIR / "val.en"
VAL_FR = DATA_DIR / "val.fr"
TEST_EN = DATA_DIR / "test.en"
TEST_FR = DATA_DIR / "test.fr"
```

**Giải thích:**
- `Path()` object: Tự động xử lý dấu `/` hoặc `\` tùy hệ điều hành
- `TRAIN_EN`: Đường dẫn tới file train.en
- `TRAIN_FR`: File train.fr tương ứng (cùng số dòng với train.en)

**Ví dụ:**
```
train.en dòng 1: "A man is walking."
train.fr dòng 1: "Un homme marche."
```

---

#### **1.3. Vocabulary Configuration**

```python
MAX_VOCAB_SIZE = 10000
MIN_FREQ = 1

PAD_TOKEN = "<pad>"  # Padding (độ dài không đủ)
UNK_TOKEN = "<unk>"  # Unknown word (từ không có trong vocab)
SOS_TOKEN = "<sos>"  # Start of sentence
EOS_TOKEN = "<eos>"  # End of sentence

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
```

**Giải thích:**

1. **MAX_VOCAB_SIZE = 10000**:
   - Chỉ giữ 10,000 từ phổ biến nhất
   - Từ hiếm → thay bằng `<unk>`
   - **Tại sao?** Giảm kích thước model, tăng tốc training

2. **Special Tokens**:
   ```
   <pad>: Thêm vào câu ngắn để bằng câu dài nhất trong batch
   <unk>: Thay thế từ không biết
   <sos>: Đánh dấu bắt đầu câu (Start Of Sentence)
   <eos>: Đánh dấu kết thúc câu (End Of Sentence)
   ```

3. **Ví dụ:**
   ```python
   Câu gốc: ["hello", "world"]
   Sau khi thêm: ["<sos>", "hello", "world", "<eos>"]
   Encode: [2, 245, 567, 3]
   
   Câu ngắn hơn: ["hi"]
   Thêm: ["<sos>", "hi", "<eos>", "<pad>"]
   Encode: [2, 123, 3, 0]
   ```

---

#### **1.4. Model Configuration**

```python
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
TEACHER_FORCING_RATIO = 0.5
```

**Giải thích:**

1. **EMBEDDING_DIM = 256**:
   - Mỗi từ → vector 256 số thực
   - Ví dụ: `"cat" → [0.23, -0.45, 0.67, ..., 0.12]` (256 số)
   - **Tại sao?** Biểu diễn ý nghĩa từ trong không gian liên tục

2. **HIDDEN_SIZE = 512**:
   - Kích thước hidden state của LSTM
   - Càng lớn → model càng mạnh, nhưng chậm hơn

3. **NUM_LAYERS = 2**:
   - LSTM 2 tầng chồng lên nhau
   ```
   Input → LSTM Layer 1 → LSTM Layer 2 → Output
   ```

4. **DROPOUT = 0.3**:
   - Tắt ngẫu nhiên 30% neurons khi training
   - **Tại sao?** Tránh overfitting

5. **TEACHER_FORCING_RATIO = 0.5**:
   - 50% thời gian: Decoder nhận từ đúng từ ground truth
   - 50% thời gian: Decoder nhận từ dự đoán của chính nó
   
   **Ví dụ:**
   ```
   Ground truth: "le chat dort"
   
   Bước 1: Input <sos> → Predict "le"
   Bước 2 (teacher forcing): Input "le" (đúng) → Predict "chat"
   Bước 3 (no teacher forcing): Input "chien" (sai) → Predict "dort"
   ```

---

#### **1.5. Training Configuration**

```python
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 64

EARLY_STOPPING_PATIENCE = 3
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
```

**Giải thích:**

1. **NUM_EPOCHS = 15**:
   - Model sẽ xem toàn bộ data 15 lần
   - 1 epoch = 454 batches (29,000 / 64)

2. **LEARNING_RATE = 0.001**:
   - Bước nhảy khi cập nhật weights
   - Quá lớn → không hội tụ
   - Quá nhỏ → học chậm

3. **BATCH_SIZE = 64**:
   - Xử lý 64 câu cùng lúc
   - Lớn hơn → nhanh hơn, nhưng tốn RAM/VRAM

4. **EARLY_STOPPING_PATIENCE = 3**:
   ```
   Epoch 5: val_loss = 3.2 ✅ (best)
   Epoch 6: val_loss = 3.3 (không giảm, patience = 1)
   Epoch 7: val_loss = 3.4 (không giảm, patience = 2)
   Epoch 8: val_loss = 3.5 (không giảm, patience = 3)
   → DỪNG! Tránh overfitting
   ```

5. **SCHEDULER_PATIENCE = 2, FACTOR = 0.5**:
   ```
   Val loss không giảm sau 2 epochs
   → Learning rate = 0.001 * 0.5 = 0.0005
   ```

---

#### **1.6. Device Configuration**

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Giải thích:**
- `cuda`: GPU (nhanh ~10-20 lần CPU)
- `cpu`: CPU (chậm, nhưng luôn có)
- **Auto detect**: Nếu có GPU thì dùng, không thì dùng CPU

**Kiểm tra:**
```python
print(DEVICE)
# Output: cuda (nếu có GPU)
# Output: cpu (nếu không có GPU)
```

---

## 2. utils.py

### 🎯 Mục đích
Chứa các hàm tiện ích: Vocabulary class, tokenization, đọc file, lưu/load vocab.

---

### 📝 Chi tiết từng class/function

#### **2.1. Class Vocabulary**

```python
class Vocabulary:
    def __init__(self, max_size=10000, min_freq=1, special_tokens=None):
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<sos>", "<eos>"]
        
        self.token2idx = {}  # word → index
        self.idx2token = {}  # index → word
        
        # Khởi tạo special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
```

**Giải thích:**
- `token2idx`: Dictionary mapping từ → số
  ```python
  {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3, "hello": 4, "world": 5}
  ```
- `idx2token`: Dictionary mapping số → từ
  ```python
  {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>", 4: "hello", 5: "world"}
  ```

---

#### **2.2. Vocabulary.build_vocab_from_iterator()**

```python
def build_vocab_from_iterator(self, iterator):
    # Đếm tần suất
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    
    # Loại bỏ special tokens nếu có trong data
    for special in self.special_tokens:
        if special in counter:
            del counter[special]
    
    # Lấy top max_size - 4 từ phổ biến nhất
    most_common = counter.most_common(self.max_size - len(self.special_tokens))
    
    # Thêm vào vocabulary
    for idx, (token, freq) in enumerate(most_common, start=len(self.special_tokens)):
        if freq >= self.min_freq:
            self.token2idx[token] = idx
            self.idx2token[idx] = token
```

**Giải thích từng bước:**

**Bước 1: Đếm tần suất**
```python
Input: [["hello", "world"], ["hello", "hi"], ["world", "peace"]]

Counter: {"hello": 2, "world": 2, "hi": 1, "peace": 1}
```

**Bước 2: Sắp xếp theo tần suất**
```python
most_common = [("hello", 2), ("world", 2), ("hi", 1), ("peace", 1)]
```

**Bước 3: Lấy top 10,000 - 4 = 9,996 từ**
```python
# Giả sử max_size = 10
# Đã có 4 special tokens
# → Chỉ lấy 6 từ tiếp theo

most_common[:6]
```

**Bước 4: Tạo mapping**
```python
token2idx = {
    "<pad>": 0,
    "<unk>": 1,
    "<sos>": 2,
    "<eos>": 3,
    "hello": 4,   # Từ phổ biến nhất
    "world": 5,   # Từ phổ biến thứ 2
    ...
}
```

---

#### **2.3. Vocabulary.encode() và decode()**

```python
def encode(self, tokens: List[str]) -> List[int]:
    return [self.token2idx.get(token, self.unk_idx) for token in tokens]

def decode(self, indices: List[int]) -> List[str]:
    return [self.idx2token.get(idx, "<unk>") for idx in indices]
```

**Ví dụ:**

```python
vocab = Vocabulary()
vocab.token2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3, "hello": 4}

# Encode
tokens = ["<sos>", "hello", "xyz", "<eos>"]
indices = vocab.encode(tokens)
# Output: [2, 4, 1, 3]
# "xyz" không có trong vocab → 1 (<unk>)

# Decode
indices = [2, 4, 1, 3]
tokens = vocab.decode(indices)
# Output: ["<sos>", "hello", "<unk>", "<eos>"]
```

---

#### **2.4. tokenize_sentence()**

```python
def tokenize_sentence(sentence: str, language: str = "en") -> List[str]:
    # Lowercase
    sentence = sentence.lower()
    
    # Thêm space trước dấu câu
    sentence = re.sub(r"([.!?;,])", r" \1", sentence)
    
    # Split by whitespace
    tokens = sentence.split()
    
    return tokens
```

**Ví dụ:**

```python
Input: "Hello, how are you?"

Bước 1: Lowercase
→ "hello, how are you?"

Bước 2: Thêm space trước dấu câu
→ "hello , how are you ?"

Bước 3: Split
→ ["hello", ",", "how", "are", "you", "?"]
```

**Tại sao tách dấu câu?**
- Mỗi dấu câu là 1 token riêng
- Model học được ý nghĩa dấu câu (câu hỏi, ngạc nhiên...)

---

#### **2.5. read_parallel_corpus()**

```python
def read_parallel_corpus(src_file: str, tgt_file: str, tokenize_fn=tokenize_sentence):
    src_sentences = []
    tgt_sentences = []
    
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            
            if src_line and tgt_line:
                src_tokens = tokenize_fn(src_line, language="en")
                tgt_tokens = tokenize_fn(tgt_line, language="fr")
                
                src_sentences.append(src_tokens)
                tgt_sentences.append(tgt_tokens)
    
    return src_sentences, tgt_sentences
```

**Giải thích:**

```python
File train.en:
  Dòng 1: "A man is walking."
  Dòng 2: "The cat is sleeping."

File train.fr:
  Dòng 1: "Un homme marche."
  Dòng 2: "Le chat dort."

Sau khi đọc:
src_sentences = [
    ["a", "man", "is", "walking", "."],
    ["the", "cat", "is", "sleeping", "."]
]

tgt_sentences = [
    ["un", "homme", "marche", "."],
    ["le", "chat", "dort", "."]
]
```

**Tại sao dùng `zip()`?**
- Đọc 2 file song song, đảm bảo dòng 1 file A tương ứng dòng 1 file B

---

#### **2.6. add_special_tokens()**

```python
def add_special_tokens(tokens: List[str], add_sos=True, add_eos=True) -> List[str]:
    result = tokens.copy()
    if add_sos:
        result = ["<sos>"] + result
    if add_eos:
        result = result + ["<eos>"]
    return result
```

**Ví dụ:**

```python
Input: ["hello", "world"]

add_special_tokens(tokens, add_sos=True, add_eos=True)
→ ["<sos>", "hello", "world", "<eos>"]

add_special_tokens(tokens, add_sos=False, add_eos=True)
→ ["hello", "world", "<eos>"]
```

**Tại sao cần?**
- `<sos>`: Báo cho Decoder biết bắt đầu sinh câu
- `<eos>`: Báo cho Decoder biết dừng lại

---

#### **2.7. save_vocab() và load_vocab()**

```python
def save_vocab(vocab, filepath):
    torch.save({
        'token2idx': vocab.token2idx,
        'idx2token': vocab.idx2token,
        'max_size': vocab.max_size,
        'min_freq': vocab.min_freq,
        'special_tokens': vocab.special_tokens
    }, filepath)
```

**Giải thích:**
- Lưu toàn bộ thông tin vocab vào file `.pth`
- **Tại sao cần?** Không phải build lại vocab mỗi lần chạy

```python
# Lưu
save_vocab(src_vocab, "src_vocab.pth")

# Load
src_vocab = load_vocab("src_vocab.pth")
```

---

## 3. data_loader.py

### 🎯 Mục đích
Xử lý dữ liệu thành batch, sắp xếp, padding để đưa vào model.

---

### 📝 Chi tiết từng class/function

#### **3.1. Class TranslationDataset**

```python
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences):
        assert len(src_sentences) == len(tgt_sentences)
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]
```

**Giải thích:**
- PyTorch Dataset wrapper
- `__getitem__`: Trả về 1 cặp câu (EN, FR) tại index `idx`

**Ví dụ:**

```python
src = [["hello", "world"], ["hi", "there"]]
tgt = [["bonjour", "monde"], ["salut", "là"]]

dataset = TranslationDataset(src, tgt)

print(len(dataset))  # 2
print(dataset[0])    # (["hello", "world"], ["bonjour", "monde"])
```

---

#### **3.2. collate_batch_with_packing()**

**Đây là hàm QUAN TRỌNG NHẤT!** Xử lý 1 batch data.

```python
def collate_batch_with_packing(batch, src_vocab, tgt_vocab, device, max_len=50):
    # Bước 1: Thêm special tokens và encode
    batch_data = []
    for src_tokens, tgt_tokens in batch:
        src_tokens = add_special_tokens(src_tokens[:max_len-2], add_sos=True, add_eos=True)
        tgt_tokens = add_special_tokens(tgt_tokens[:max_len-2], add_sos=True, add_eos=True)
        
        src_indices = src_vocab.encode(src_tokens)
        tgt_indices = tgt_vocab.encode(tgt_tokens)
        
        batch_data.append((src_indices, len(src_indices), tgt_indices, len(tgt_indices)))
    
    # Bước 2: Sắp xếp theo độ dài giảm dần
    batch_data.sort(key=lambda x: x[1], reverse=True)
    
    # Bước 3: Padding
    # ... (chi tiết bên dưới)
```

**Giải thích từng bước:**

### **Bước 1: Encode sentences**

```python
Input batch (2 câu):
[
    (["hello", "world"], ["bonjour", "monde"]),
    (["hi"], ["salut"])
]

Sau khi thêm special tokens:
[
    (["<sos>", "hello", "world", "<eos>"], ["<sos>", "bonjour", "monde", "<eos>"]),
    (["<sos>", "hi", "<eos>"], ["<sos>", "salut", "<eos>"])
]

Sau khi encode:
[
    ([2, 523, 890, 3], 4, [2, 312, 567, 3], 4),
    ([2, 124, 3], 3, [2, 234, 3], 3)
]
```

### **Bước 2: Sắp xếp theo độ dài giảm dần**

```python
Trước sắp xếp:
[
    ([2, 523, 890, 3], 4, ...),  # Độ dài 4
    ([2, 124, 3], 3, ...)        # Độ dài 3
]

Sau sắp xếp:
[
    ([2, 523, 890, 3], 4, ...),  # Câu dài nhất lên đầu
    ([2, 124, 3], 3, ...)        # Câu ngắn hơn xuống dưới
]
```

**Tại sao phải sắp xếp?**
- `pack_padded_sequence` yêu cầu batch phải sắp xếp giảm dần
- Giúp LSTM xử lý hiệu quả hơn (bỏ qua padding tokens)

### **Bước 3: Padding**

```python
max_src_len = 4
max_tgt_len = 4

Padding source:
[2, 523, 890, 3]       → [2, 523, 890, 3]      (đủ dài)
[2, 124, 3]            → [2, 124, 3, 0]        (thêm 1 padding)

Padding target:
[2, 312, 567, 3]       → [2, 312, 567, 3]
[2, 234, 3]            → [2, 234, 3, 0]
```

### **Bước 4: Chuyển sang tensor**

```python
src_batch = torch.tensor([
    [2, 523, 890, 3],
    [2, 124, 3, 0]
], device='cuda')

src_lengths = torch.tensor([4, 3], device='cpu')  # Phải CPU!
```

**Tại sao lengths phải ở CPU?**
- `pack_padded_sequence` yêu cầu lengths ở CPU
- PyTorch bug nếu để GPU

---

#### **3.3. build_vocabularies()**

```python
def build_vocabularies(train_src_file, train_tgt_file, max_vocab_size=10000):
    # Đọc training data
    src_sentences, tgt_sentences = read_parallel_corpus(
        train_src_file, 
        train_tgt_file,
        tokenize_fn=tokenize_sentence
    )
    
    # Build source vocabulary
    src_vocab = Vocabulary(max_size=max_vocab_size, ...)
    src_vocab.build_vocab_from_iterator(src_sentences)
    
    # Build target vocabulary
    tgt_vocab = Vocabulary(max_size=max_vocab_size, ...)
    tgt_vocab.build_vocab_from_iterator(tgt_sentences)
    
    return src_vocab, tgt_vocab
```

**Giải thích:**
1. Đọc file train.en và train.fr
2. Tokenize tất cả câu
3. Build 2 vocabularies riêng biệt (EN và FR)
4. Trả về 2 vocab objects

**Tại sao build 2 vocab riêng?**
- Tiếng Anh và tiếng Pháp có từ vựng khác nhau
- Mỗi ngôn ngữ 10,000 tokens riêng

---

#### **3.4. prepare_data_loaders()**

```python
def prepare_data_loaders(src_vocab, tgt_vocab, batch_size=64):
    # Load data
    train_src, train_tgt = read_parallel_corpus(TRAIN_EN, TRAIN_FR)
    train_dataset = TranslationDataset(train_src, train_tgt)
    
    # Create collate function
    def collate_fn_wrapper(batch):
        return collate_batch_with_packing(
            batch, src_vocab, tgt_vocab, DEVICE, MAX_SEQ_LENGTH
        )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_wrapper,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader
```

**Giải thích:**

1. **Load data**: Đọc file và tạo Dataset
2. **collate_fn_wrapper**: Wrapper để truyền thêm tham số vào collate function
3. **DataLoader**: PyTorch DataLoader
   - `shuffle=True`: Trộn ngẫu nhiên data (chỉ cho train)
   - `collate_fn`: Hàm xử lý batch
   - `pin_memory=False`: Không pin memory (vì đã chuyển lên GPU trong collate_fn)

**Output:**

```python
train_loader: 454 batches (29,000 / 64)
val_loader: 16 batches (1,014 / 64)
test_loader: 16 batches (1,000 / 64)
```

---

## 4. Luồng hoạt động tổng thể

### 🔄 **Từ file data → Model input**

```
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 1: ĐỌC FILE                                            │
└─────────────────────────────────────────────────────────────┘
train.en: "A man is walking in the street."
train.fr: "Un homme marche dans la rue."
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 2: TOKENIZATION                                        │
└─────────────────────────────────────────────────────────────┘
EN: ["a", "man", "is", "walking", "in", "the", "street", "."]
FR: ["un", "homme", "marche", "dans", "la", "rue", "."]
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 3: BUILD VOCABULARY (chỉ chạy 1 lần)                  │
└─────────────────────────────────────────────────────────────┘
src_vocab: {"<pad>":0, "<unk>":1, ..., "man":523, "walking":1247}
tgt_vocab: {"<pad>":0, "<unk>":1, ..., "homme":312, "marche":456}
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 4: ADD SPECIAL TOKENS                                  │
└─────────────────────────────────────────────────────────────┘
EN: ["<sos>", "a", "man", "is", "walking", ..., ".", "<eos>"]
FR: ["<sos>", "un", "homme", "marche", ..., ".", "<eos>"]
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 5: ENCODE (word → index)                               │
└─────────────────────────────────────────────────────────────┘
EN: [2, 12, 523, 45, 1247, 89, 12, 678, 5, 3]
FR: [2, 312, 456, 234, 67, 445, 5, 3]
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 6: TẠO BATCH (64 câu)                                  │
└─────────────────────────────────────────────────────────────┘
batch = [
    (câu_1_EN, câu_1_FR),
    (câu_2_EN, câu_2_FR),
    ...
    (câu_64_EN, câu_64_FR)
]
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 7: SẮP XẾP THEO ĐỘ DÀI GIẢM DẦN                       │
└─────────────────────────────────────────────────────────────┘
Câu dài nhất lên đầu, câu ngắn nhất xuống cuối
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 8: PADDING                                             │
└─────────────────────────────────────────────────────────────┘
Thêm <pad> (0) vào câu ngắn để bằng câu dài nhất
          ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 9: CHUYỂN SANG TENSOR GPU                              │
└─────────────────────────────────────────────────────────────┘
src_batch: torch.tensor([[...], [...], ...], device='cuda')
src_lengths: torch.tensor([25, 23, 20, ...], device='cpu')
tgt_batch: torch.tensor([[...], [...], ...], device='cuda')
tgt_lengths: torch.tensor([28, 25, 22, ...], device='cpu')
          ↓
┌─────────────────────────────────────────────────────────────┐
│ SẴN SÀNG CHO MODEL!                                         │
└─────────────────────────────────────────────────────────────┘
```

---

### 🧠 **Trong quá trình training**

```python
for epoch in range(NUM_EPOCHS):
    for src_batch, src_lengths, tgt_batch, tgt_lengths in train_loader:
        # src_batch: (64, 25) - 64 câu, mỗi câu tối đa 25 tokens
        # src_lengths: (64,) - [25, 23, 20, ..., 8]
        # tgt_batch: (64, 28) - 64 câu tiếng Pháp
        # tgt_lengths: (64,) - [28, 25, 22, ..., 10]
        
        # Forward pass
        output = model(src_batch, src_lengths, tgt_batch)
        
        # Compute loss
        loss = criterion(output, tgt_batch)
        
        # Backward + Update
        loss.backward()
        optimizer.step()
```

---

## 5. Câu hỏi thường gặp

### ❓ **Tại sao phải sắp xếp batch theo độ dài giảm dần?**

**Trả lời:**
- `pack_padded_sequence` **yêu cầu bắt buộc** batch phải sắp xếp giảm dần
- Nếu không sắp xếp → lỗi runtime

**Lợi ích:**
```python
Không có packing:
Câu 1: [1, 2, 3, 4, 5, 0, 0, 0]  # LSTM xử lý cả 8 tokens (lãng phí)
Câu 2: [6, 7, 8, 0, 0, 0, 0, 0]  # LSTM xử lý cả 8 tokens (lãng phí)

Có packing:
Câu 1: [1, 2, 3, 4, 5]  # LSTM chỉ xử lý 5 tokens (hiệu quả)
Câu 2: [6, 7, 8]        # LSTM chỉ xử lý 3 tokens (hiệu quả)
```

---

### ❓ **Tại sao cần special tokens?**

**Trả lời:**

1. **`<pad>` (padding)**:
   - Câu ngắn cần padding để bằng câu dài nhất
   - Model học bỏ qua padding (không tính loss)

2. **`<unk>` (unknown)**:
   - Từ không có trong vocab (từ hiếm)
   - Thay bằng `<unk>` thay vì báo lỗi

3. **`<sos>` (start of sentence)**:
   - Decoder cần biết bắt đầu từ đâu
   - Input đầu tiên của Decoder luôn là `<sos>`

4. **`<eos>` (end of sentence)**:
   - Decoder biết khi nào dừng sinh từ
   - Khi predict `<eos>` → dừng lại

**Ví dụ:**
```python
Decoder:
Input: <sos> → Predict: "le"
Input: "le" → Predict: "chat"
Input: "chat" → Predict: "dort"
Input: "dort" → Predict: <eos>
→ Dừng! Output: "le chat dort"
```

---

### ❓ **Tại sao lengths phải ở CPU?**

**Trả lời:**
- Bug của PyTorch: `pack_padded_sequence` yêu cầu lengths ở CPU
- Nếu để GPU → lỗi runtime

```python
src_lengths = torch.tensor(src_lengths, device='cpu')  # ✅ Đúng
src_lengths = torch.tensor(src_lengths, device='cuda')  # ❌ Lỗi
```

---

### ❓ **Tại sao pin_memory=False?**

**Trả lời:**
- `pin_memory=True`: Dùng khi tensor ở CPU, muốn chuyển nhanh lên GPU
- Nhưng trong `collate_batch_with_packing`, tensor đã ở GPU rồi
- PyTorch không thể pin tensor GPU → lỗi

```python
# Trong collate_fn:
src_batch = torch.tensor(..., device='cuda')  # Đã ở GPU

# Trong DataLoader:
pin_memory=False  # Phải False vì tensor đã ở GPU
```

---

### ❓ **Teacher forcing là gì?**

**Trả lời:**
Kỹ thuật training Decoder: đôi khi cho Decoder nhận từ đúng, đôi khi cho nhận từ dự đoán.

**Ví dụ:**
```python
Ground truth: "le chat dort"

Bước 1: Input <sos> → Predict "le"

Bước 2 (teacher_forcing=True):
  Input: "le" (từ ground truth) → Predict "chat"

Bước 3 (teacher_forcing=False):
  Input: "chien" (từ dự đoán sai) → Predict "dort"
```

**Tại sao cần?**
- Teacher forcing = 1.0: Model học nhanh, nhưng không robust
- Teacher forcing = 0.0: Model học chậm, nhưng robust hơn
- Teacher forcing = 0.5: Cân bằng giữa 2 cái

---

### ❓ **Tại sao MAX_VOCAB_SIZE = 10,000?**

**Trả lời:**

1. **Từ hiếm xuất hiện ít, không quan trọng**:
   ```
   "cat": 1,000 lần
   "dog": 800 lần
   "supercalifragilisticexpialidocious": 1 lần → Loại bỏ
   ```

2. **Giảm kích thước model**:
   ```
   Vocab 50,000: Embedding = 50,000 x 256 = 12.8M params
   Vocab 10,000: Embedding = 10,000 x 256 = 2.56M params
   → Nhẹ hơn 5 lần!
   ```

3. **Training nhanh hơn**:
   - Vocab nhỏ → softmax nhanh hơn
   - Vocab 10,000: ~10ms/batch
   - Vocab 50,000: ~50ms/batch

---

### ❓ **Batch size lớn hay nhỏ?**

**So sánh:**

| Batch Size | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| **32** | Ổn định, ít RAM | Chậm (nhiều iteration) |
| **64** | Cân bằng (khuyến nghị) | - |
| **128** | Nhanh, gradient ổn định | Tốn RAM/VRAM |
| **256** | Rất nhanh | Có thể không fit GPU |

**Khuyến nghị:**
- GPU T4 (Colab): Batch size = 64-128
- GPU V100: Batch size = 128-256
- CPU: Batch size = 32

---

## 📚 TÓM TẮT

### 📦 **3 file code chính:**

1. **config.py**: Cấu hình (hyperparameters, paths, device)
2. **utils.py**: Vocabulary, tokenization, đọc file
3. **data_loader.py**: Dataset, DataLoader, collate function

### 🔄 **Quy trình:**

```
File → Tokenize → Build Vocab → Encode → Batch → Sort → Pad → Tensor → Model
```

### 🎯 **Các khái niệm quan trọng:**

- **Vocabulary**: Mapping word ↔ index
- **Special tokens**: `<pad>`, `<unk>`, `<sos>`, `<eos>`
- **Batch**: Nhóm 64 câu xử lý cùng lúc
- **Padding**: Thêm `<pad>` để câu ngắn bằng câu dài
- **Sorting**: Sắp xếp giảm dần để dùng `pack_padded_sequence`
- **Teacher forcing**: Đôi khi cho Decoder nhận từ đúng

---

**Hy vọng giải thích này giúp bạn hiểu rõ toàn bộ code!** 🚀

*Nếu còn câu hỏi gì, cứ hỏi tiếp!*
