"""
Configuration file cho đồ án NLP - Encoder-Decoder LSTM
Task 1: Thiết lập môi trường + cấu trúc project
"""

import torch
from pathlib import Path
import os

# ============ PATH CONFIGURATION ============
# Tương thích cả local và Colab
try:
    # Nếu chạy từ file .py (local)
    BASE_DIR = Path(__file__).parent.parent
except NameError:
    # Nếu chạy trên Colab/Jupyter (không có __file__)
    BASE_DIR = Path("/content")

DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "check_point"
REPORT_DIR = BASE_DIR / "report"

# Tạo thư mục nếu chưa có
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)

# ============ DATA FILES ============
TRAIN_EN = DATA_DIR / "train.en"
TRAIN_FR = DATA_DIR / "train.fr"
VAL_EN = DATA_DIR / "val.en"
VAL_FR = DATA_DIR / "val.fr"
TEST_EN = DATA_DIR / "test.en"
TEST_FR = DATA_DIR / "test.fr"

# ============ VOCABULARY CONFIGURATION ============
# Theo yêu cầu: "Giới hạn 10,000 từ phổ biến nhất mỗi ngôn ngữ"
MAX_VOCAB_SIZE = 10000
MIN_FREQ = 1  # Tần suất tối thiểu để từ được đưa vào vocab

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"  # Start of sentence
EOS_TOKEN = "<eos>"  # End of sentence

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

# Token indices
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# ============ DATA PROCESSING ============
# Batch size: 32-128 theo yêu cầu
BATCH_SIZE = 64  # Có thể thay đổi trong khoảng [32, 128]
MAX_SEQ_LENGTH = 50  # Giới hạn độ dài câu để tránh quá dài

# Sorting & Packing (yêu cầu Task 2)
SORT_WITHIN_BATCH = True  # Sắp xếp batch theo độ dài giảm dần
USE_PACKED_SEQUENCE = True  # Sử dụng pack_padded_sequence

# ============ MODEL CONFIGURATION ============
# Theo bảng "Tham số khuyến nghị" trong đề bài
EMBEDDING_DIM = 256  # Có thể từ 256-512
HIDDEN_SIZE = 512
NUM_LAYERS = 2  # Số layer LSTM
DROPOUT = 0.3  # Từ 0.3-0.5 theo đề bài
TEACHER_FORCING_RATIO = 0.5

# Encoder-Decoder với context vector cố định (không dùng attention)
USE_ATTENTION = False  # Task cơ bản không yêu cầu attention

# ============ TRAINING CONFIGURATION ============
NUM_EPOCHS = 15  # Từ 10-20 theo đề bài
LEARNING_RATE = 0.001
OPTIMIZER = "Adam"  # Adam(lr=0.001) theo đề bài

# Scheduler: ReduceLROnPlateau
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 3  # Dừng nếu val_loss không giảm sau 3 epochs

# Checkpoint
SAVE_BEST_MODEL = True
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

# ============ DEVICE CONFIGURATION ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ EVALUATION ============
# Greedy decoding: max length = 50 hoặc gặp <eos>
MAX_DECODE_LENGTH = 50

# BLEU score
COMPUTE_BLEU = True

# ============ LOGGING ============
PRINT_EVERY = 100  # Print loss mỗi 100 batches
SAVE_PLOTS = True  # Lưu biểu đồ train/val loss

# ============ DISPLAY CONFIG ============
def display_config():
    """In ra cấu hình hiện tại"""
    print("=" * 60)
    print("CẤU HÌNH - Dịch Anh-Pháp")
    print("=" * 60)
    print(f"Thiết bị: {DEVICE}")
    print(f"Kích thước batch: {BATCH_SIZE}")
    print(f"Kích thước từ điển tối đa: {MAX_VOCAB_SIZE}")
    print(f"Chiều embedding: {EMBEDDING_DIM}")
    print(f"Kích thước hidden: {HIDDEN_SIZE}")
    print(f"Số lớp: {NUM_LAYERS}")
    print(f"Dropout: {DROPOUT}")
    print(f"Tỉ lệ teacher forcing: {TEACHER_FORCING_RATIO}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Số epoch: {NUM_EPOCHS}")
    print(f"Độ kiên nhẫn early stopping: {EARLY_STOPPING_PATIENCE}")
    print("=" * 60)

if __name__ == "__main__":
    display_config()
