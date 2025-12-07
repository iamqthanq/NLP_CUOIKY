"""
Task 2: Data Processing & DataLoader
Xử lý dữ liệu, tokenization, vocabulary building, padding/packing
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import Counter
from typing import List, Tuple
import re

# NOTE: Trong Colab, các biến này đã được define ở cell Config
# Nếu chạy file này standalone, cần uncomment các dòng dưới:

# from pathlib import Path
# DATA_DIR = Path("/content/data")
# CHECKPOINT_DIR = Path("/content/check_point")
# TRAIN_EN = DATA_DIR / "train.en"
# TRAIN_FR = DATA_DIR / "train.fr"
# VAL_EN = DATA_DIR / "val.en"
# VAL_FR = DATA_DIR / "val.fr"
# TEST_EN = DATA_DIR / "test.en"
# TEST_FR = DATA_DIR / "test.fr"
# BATCH_SIZE = 64
# MAX_SEQ_LENGTH = 50
# MAX_VOCAB_SIZE = 10000
# MIN_FREQ = 1
# SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranslationDataset(Dataset):
    """
    Dataset cho bài toán Machine Translation
    """
    def __init__(self, src_sentences, tgt_sentences):
        """
        Args:
            src_sentences: List of tokenized source sentences
            tgt_sentences: List of tokenized target sentences
        """
        assert len(src_sentences) == len(tgt_sentences), \
            "Source and target must have same length"
        
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]


def collate_batch_with_packing(batch, src_vocab, tgt_vocab, device, max_len=50):
    """
    Collate function với sorting và packing
    Yêu cầu: Sắp xếp batch theo độ dài giảm dần, sử dụng pack_padded_sequence
    """
    # Thêm special tokens và encode
    batch_data = []
    for src_tokens, tgt_tokens in batch:
        # Giới hạn độ dài và thêm <sos>, <eos>
        src_tokens = add_special_tokens(src_tokens[:max_len-2], add_sos=True, add_eos=True)
        tgt_tokens = add_special_tokens(tgt_tokens[:max_len-2], add_sos=True, add_eos=True)
        
        # Encode to indices
        src_indices = src_vocab.encode(src_tokens)
        tgt_indices = tgt_vocab.encode(tgt_tokens)
        
        batch_data.append((src_indices, len(src_indices), tgt_indices, len(tgt_indices)))
    
    # Sort by source length (descending) - yêu cầu cho pack_padded_sequence
    batch_data.sort(key=lambda x: x[1], reverse=True)
    
    # Unpack
    src_batch = [item[0] for item in batch_data]
    src_lengths = [item[1] for item in batch_data]
    tgt_batch = [item[2] for item in batch_data]
    tgt_lengths = [item[3] for item in batch_data]
    
    # Padding
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    
    padded_src = []
    padded_tgt = []
    
    for src_indices, tgt_indices in zip(src_batch, tgt_batch):
        padded_src.append(src_indices + [src_vocab.pad_idx] * (max_src_len - len(src_indices)))
        padded_tgt.append(tgt_indices + [tgt_vocab.pad_idx] * (max_tgt_len - len(tgt_indices)))
    
    # Convert to tensors
    src_batch = torch.tensor(padded_src, dtype=torch.long, device=device)
    tgt_batch = torch.tensor(padded_tgt, dtype=torch.long, device=device)
    src_lengths = torch.tensor(src_lengths, dtype=torch.long, device='cpu')  # lengths phải ở CPU
    tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long, device='cpu')
    
    return src_batch, src_lengths, tgt_batch, tgt_lengths


def build_vocabularies(train_src_file, train_tgt_file, max_vocab_size=10000):
    """
    Xây dựng vocabularies từ training data
    
    """
    print("=" * 60)
    print("XÂY DỰNG TỪ ĐIỂN (VOCABULARIES)")
    print("=" * 60)
    
    # Đọc training data
    print(f"Đọc dữ liệu huấn luyện từ:")
    print(f"  Source: {train_src_file}")
    print(f"  Target: {train_tgt_file}")
    
    src_sentences, tgt_sentences = read_parallel_corpus(
        train_src_file, 
        train_tgt_file,
        tokenize_fn=tokenize_sentence
    )
    
    print(f"Đã tải {len(src_sentences)} cặp câu")
    
    # Build source vocabulary
    print("\nXây dựng từ điển tiếng Anh (source)...")
    src_vocab = Vocabulary(
        max_size=max_vocab_size,
        min_freq=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS
    )
    src_vocab.build_vocab_from_iterator(src_sentences)
    
    # Build target vocabulary
    print("\nXây dựng từ điển tiếng Pháp (target)...")
    tgt_vocab = Vocabulary(
        max_size=max_vocab_size,
        min_freq=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS
    )
    tgt_vocab.build_vocab_from_iterator(tgt_sentences)
    
    print("=" * 60)
    print(f"Kích thước từ điển tiếng Anh: {len(src_vocab)}")
    print(f"Kích thước từ điển tiếng Pháp: {len(tgt_vocab)}")
    print("=" * 60)
    
    return src_vocab, tgt_vocab


def prepare_data_loaders(src_vocab, tgt_vocab, batch_size=64):
    """
    Chuẩn bị DataLoaders cho train, val, test
    
    Args:
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        batch_size: Batch size (32-128 theo đề bài)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("=" * 60)
    print("CHUẨN BỊ DATA LOADERS")
    print("=" * 60)
    
    # Load train data
    print("Đang tải dữ liệu huấn luyện...")
    train_src, train_tgt = read_parallel_corpus(TRAIN_EN, TRAIN_FR)
    train_dataset = TranslationDataset(train_src, train_tgt)
    print(f"  Kích thước tập train: {len(train_dataset)}")
    
    # Load val data
    print("Đang tải dữ liệu validation...")
    val_src, val_tgt = read_parallel_corpus(VAL_EN, VAL_FR)
    val_dataset = TranslationDataset(val_src, val_tgt)
    print(f"  Kích thước tập val: {len(val_dataset)}")
    
    # Load test data
    print("Đang tải dữ liệu test...")
    test_src, test_tgt = read_parallel_corpus(TEST_EN, TEST_FR)
    test_dataset = TranslationDataset(test_src, test_tgt)
    print(f"  Kích thước tập test: {len(test_dataset)}")
    
    # Create collate function
    def collate_fn_wrapper(batch):
        return collate_batch_with_packing(
            batch, src_vocab, tgt_vocab, DEVICE, MAX_SEQ_LENGTH
        )
    
    # Create DataLoaders
    # enforce_sorted=True để sử dụng pack_padded_sequence
    # NOTE: pin_memory=False vì tensor đã được chuyển lên GPU trong collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        collate_fn=collate_fn_wrapper,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        pin_memory=False
    )
    
    print(f"\nKích thước batch: {batch_size}")
    print(f"Số batch train: {len(train_loader)}")
    print(f"Số batch val: {len(val_loader)}")
    print(f"Số batch test: {len(test_loader)}")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader


def test_data_loading():
    """
    Test function để kiểm tra data loading
    """
    print("Kiểm tra tải dữ liệu...")
    
    # Build vocabularies
    src_vocab, tgt_vocab = build_vocabularies(TRAIN_EN, TRAIN_FR, MAX_VOCAB_SIZE)
    
    # Save vocabularies
    save_vocab(src_vocab, CHECKPOINT_DIR / "src_vocab.pth")
    save_vocab(tgt_vocab, CHECKPOINT_DIR / "tgt_vocab.pth")
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        src_vocab, tgt_vocab, BATCH_SIZE
    )
    
    # Test một batch
    print("\n" + "=" * 60)
    print("KIỂM TRA MỘT BATCH")
    print("=" * 60)
    
    for src_batch, src_lengths, tgt_batch, tgt_lengths in train_loader:
        print(f"Kích thước source batch: {src_batch.shape}")
        print(f"Kích thước source lengths: {src_lengths.shape}")
        print(f"Kích thước target batch: {tgt_batch.shape}")
        print(f"Kích thước target lengths: {tgt_lengths.shape}")
        
        print(f"\nĐộ dài source (đã sắp xếp): {src_lengths.tolist()[:5]}...")
        print(f"Độ dài target: {tgt_lengths.tolist()[:5]}...")
        
        # Decode first sentence
        print("\n--- Câu đầu tiên trong batch ---")
        src_tokens = src_vocab.decode(src_batch[0].tolist())
        tgt_tokens = tgt_vocab.decode(tgt_batch[0].tolist())
        
        print(f"Source (EN): {' '.join(src_tokens[:src_lengths[0]])}")
        print(f"Target (FR): {' '.join(tgt_tokens[:tgt_lengths[0]])}")
        
        break
    
    print("\n✅ Kiểm tra tải dữ liệu hoàn tất!")


if __name__ == "__main__":
    # Run test
    test_data_loading()
