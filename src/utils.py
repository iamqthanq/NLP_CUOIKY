"""
Utility functions cho đồ án NLP
Task 1 & 2: Xử lý dữ liệu, tokenization, vocabulary building
"""

import torch
from collections import Counter
from typing import List, Tuple, Dict
import re

class Vocabulary:
    """
    Class quản lý vocabulary cho một ngôn ngữ
    Yêu cầu: Giới hạn 10,000 từ phổ biến nhất
    """
    def __init__(self, max_size=10000, min_freq=1, special_tokens=None):
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<sos>", "<eos>"]
        
        # Token to index and index to token mappings
        self.token2idx = {}
        self.idx2token = {}
        
        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        self.pad_idx = self.token2idx["<pad>"]
        self.unk_idx = self.token2idx["<unk>"]
        self.sos_idx = self.token2idx["<sos>"]
        self.eos_idx = self.token2idx["<eos>"]
    
    def build_vocab_from_iterator(self, iterator):
        """
        Xây dựng vocabulary từ iterator of sentences
        
        Args:
            iterator: Iterator chứa các câu (mỗi câu là list of tokens)
        """
        # Đếm tần suất xuất hiện của mỗi token
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)
        
        # Lọc theo min_freq và lấy max_size tokens phổ biến nhất
        # Loại bỏ special tokens nếu có trong data
        for special in self.special_tokens:
            if special in counter:
                del counter[special]
        
        # Sắp xếp theo tần suất giảm dần và lấy top max_size - len(special_tokens)
        most_common = counter.most_common(self.max_size - len(self.special_tokens))
        
        # Thêm vào vocabulary (bắt đầu từ index len(special_tokens))
        for idx, (token, freq) in enumerate(most_common, start=len(self.special_tokens)):
            if freq >= self.min_freq:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        
        print(f"Xây dựng từ điển với {len(self.token2idx)} token")
        print(f"  - Special tokens: {len(self.special_tokens)}")
        print(f"  - Token thường: {len(self.token2idx) - len(self.special_tokens)}")
    
    def __len__(self):
        return len(self.token2idx)
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx2token.get(idx, "<unk>") for idx in indices]


def tokenize_sentence(sentence: str, language: str = "en") -> List[str]:
    """
    Tokenize một câu thành list of tokens
    Sử dụng tokenization đơn giản (split by space + lowercase)
    
    Args:
        sentence: Câu cần tokenize
        language: Ngôn ngữ ('en' hoặc 'fr')
    
    Returns:
        List of tokens
    """
    # Lowercase
    sentence = sentence.lower()
    
    # Xử lý dấu câu: thêm space trước dấu câu
    sentence = re.sub(r"([.!?;,])", r" \1", sentence)
    
    # Split by whitespace
    tokens = sentence.split()
    
    return tokens


def read_parallel_corpus(src_file: str, tgt_file: str, tokenize_fn=tokenize_sentence) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Đọc parallel corpus (cặp file en-fr)
    
    Args:
        src_file: Path to source file (.en)
        tgt_file: Path to target file (.fr)
        tokenize_fn: Function để tokenize
    
    Returns:
        (src_sentences, tgt_sentences): Tuple of lists of tokenized sentences
    """
    src_sentences = []
    tgt_sentences = []
    
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            
            if src_line and tgt_line:  # Bỏ qua dòng trống
                src_tokens = tokenize_fn(src_line, language="en")
                tgt_tokens = tokenize_fn(tgt_line, language="fr")
                
                src_sentences.append(src_tokens)
                tgt_sentences.append(tgt_tokens)
    
    return src_sentences, tgt_sentences


def add_special_tokens(tokens: List[str], add_sos=True, add_eos=True) -> List[str]:
    """
    Thêm <sos> và <eos> vào câu
    
    Args:
        tokens: List of tokens
        add_sos: Thêm <sos> ở đầu
        add_eos: Thêm <eos> ở cuối
    
    Returns:
        List of tokens with special tokens
    """
    result = tokens.copy()
    if add_sos:
        result = ["<sos>"] + result
    if add_eos:
        result = result + ["<eos>"]
    return result


def collate_fn(batch, src_vocab, tgt_vocab, device, max_len=50):
    """
    Custom collate function cho DataLoader
    Xử lý padding và chuyển sang tensor
    
    Args:
        batch: List of (src_tokens, tgt_tokens)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: torch device
        max_len: Độ dài tối đa của sequence
    
    Returns:
        src_batch: Tensor shape (batch_size, max_src_len)
        src_lengths: Tensor shape (batch_size,) - độ dài thực của mỗi sequence
        tgt_batch: Tensor shape (batch_size, max_tgt_len)
        tgt_lengths: Tensor shape (batch_size,)
    """
    # Thêm special tokens
    src_batch = []
    tgt_batch = []
    src_lengths = []
    tgt_lengths = []
    
    for src_tokens, tgt_tokens in batch:
        # Thêm <sos>, <eos> và giới hạn độ dài
        src_tokens = add_special_tokens(src_tokens[:max_len-2], add_sos=True, add_eos=True)
        tgt_tokens = add_special_tokens(tgt_tokens[:max_len-2], add_sos=True, add_eos=True)
        
        # Encode
        src_indices = src_vocab.encode(src_tokens)
        tgt_indices = tgt_vocab.encode(tgt_tokens)
        
        src_batch.append(src_indices)
        tgt_batch.append(tgt_indices)
        src_lengths.append(len(src_indices))
        tgt_lengths.append(len(tgt_indices))
    
    # Padding
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    
    padded_src = []
    padded_tgt = []
    
    for src_indices, tgt_indices in zip(src_batch, tgt_batch):
        # Pad với <pad> token
        padded_src.append(src_indices + [src_vocab.pad_idx] * (max_src_len - len(src_indices)))
        padded_tgt.append(tgt_indices + [tgt_vocab.pad_idx] * (max_tgt_len - len(tgt_indices)))
    
    # Convert to tensors
    src_batch = torch.tensor(padded_src, dtype=torch.long, device=device)
    tgt_batch = torch.tensor(padded_tgt, dtype=torch.long, device=device)
    src_lengths = torch.tensor(src_lengths, dtype=torch.long, device=device)
    tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long, device=device)
    
    return src_batch, src_lengths, tgt_batch, tgt_lengths


def save_vocab(vocab, filepath):
    """Lưu vocabulary"""
    torch.save({
        'token2idx': vocab.token2idx,
        'idx2token': vocab.idx2token,
        'max_size': vocab.max_size,
        'min_freq': vocab.min_freq,
        'special_tokens': vocab.special_tokens
    }, filepath)
    print(f"Đã lưu từ điển vào {filepath}")


def load_vocab(filepath):
    """Load vocabulary"""
    data = torch.load(filepath)
    vocab = Vocabulary(
        max_size=data['max_size'],
        min_freq=data['min_freq'],
        special_tokens=data['special_tokens']
    )
    vocab.token2idx = data['token2idx']
    vocab.idx2token = data['idx2token']
    print(f"Đã tải từ điển từ {filepath}")
    return vocab


def count_parameters(model):
    """Đếm số parameters của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    """Tính thời gian chạy epoch"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    # Test tokenization
    test_sentence = "Hello, how are you?"
    tokens = tokenize_sentence(test_sentence)
    print(f"Kết quả tokenize: {tokens}")
    
    # Test vocabulary
    sentences = [
        ["hello", "world"],
        ["hello", "how", "are", "you"],
        ["world", "peace"]
    ]
    vocab = Vocabulary(max_size=100)
    vocab.build_vocab_from_iterator(sentences)
    print(f"Kích thước từ điển: {len(vocab)}")
    print(f"Mã hóa 'hello': {vocab.encode(['hello'])}")
