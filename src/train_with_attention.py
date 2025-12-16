# ============================================
# TRAINING SCRIPT WITH ATTENTION
# Mục tiêu: BLEU ~45%
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append('src')

from encoder import Encoder
from attention_decoder import Attention, AttentionDecoder, Seq2SeqWithAttention
# Import các hàm train/eval từ code cũ của bạn

# ============================================
# CONFIGURATION (Optimized for 45% BLEU)
# ============================================

CONFIG = {
    # ===== VOCABULARY (Tăng để giảm <unk>) =====
    'max_vocab_size': 15000,      # Tăng từ 10K -> 15K
    'max_seq_len': 50,
    'min_freq': 2,
    
    # ===== MODEL ARCHITECTURE =====
    'emb_dim': 256,
    'enc_hid_dim': 512,
    'dec_hid_dim': 512,
    'n_layers': 2,
    'dropout': 0.3,
    
    # ===== TRAINING =====
    'batch_size': 64,
    'num_epochs': 20,             # Tăng từ 15 -> 20
    'learning_rate': 0.001,
    'clip': 1.0,
    'teacher_forcing_ratio': 0.5,
    'early_stopping_patience': 5,  # Tăng patience
    
    # ===== DEVICE =====
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================
# BUILD MODEL WITH ATTENTION
# ============================================

def build_attention_model(src_vocab, tgt_vocab, config):
    """
    Build Seq2Seq model with Attention
    """
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    
    # Encoder
    enc = Encoder(
        INPUT_DIM,
        config['emb_dim'],
        config['enc_hid_dim'],
        config['n_layers'],
        config['dropout']
    )
    
    # Attention
    attn = Attention(config['enc_hid_dim'], config['dec_hid_dim'])
    
    # Decoder with Attention
    dec = AttentionDecoder(
        OUTPUT_DIM,
        config['emb_dim'],
        config['enc_hid_dim'],
        config['dec_hid_dim'],
        config['n_layers'],
        config['dropout'],
        attn
    )
    
    # Complete model
    model = Seq2SeqWithAttention(enc, dec, config['device'])
    
    return model

# ============================================
# TRAINING WITH LEARNING RATE SCHEDULER
# ============================================

def train_with_scheduler(model, train_iterator, val_iterator, config):
    """
    Train model with Learning Rate Scheduler
    """
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = <pad>
    
    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,      # Giảm LR xuống 50%
        patience=2,      # Sau 2 epochs không cải thiện
        verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_epoch(
            model, train_iterator, optimizer, criterion, 
            config['clip'], config['device']
        )
        
        # Evaluate
        val_loss = evaluate(model, val_iterator, criterion, config['device'])
        
        # Update learning rate based on val_loss
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'check_point/best_model_attention.pth')
            
            print(f"✅ Epoch {epoch+1}: Saved checkpoint (val_loss={val_loss:.3f})")
        else:
            patience_counter += 1
            
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"  Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        print(f"  Patience: {patience_counter}/{config['early_stopping_patience']}")
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    return model

# ============================================
# MODIFIED TRAIN/EVAL (Support Attention outputs)
# ============================================

def train_epoch(model, iterator, optimizer, criterion, clip, device):
    """Train for one epoch (modified for attention)"""
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src, src_len = batch['src'].to(device), batch['src_len'].to(device)
        trg = batch['trg'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (returns outputs AND attentions now)
        outputs, attentions = model(src, src_len, trg)
        # outputs: [batch, trg_len, output_dim]
        # attentions: [batch, trg_len, src_len] (ignore for training)
        
        # Reshape for loss
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:, :].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        # Loss
        loss = criterion(outputs, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    """Evaluate model (modified for attention)"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            src, src_len = batch['src'].to(device), batch['src_len'].to(device)
            trg = batch['trg'].to(device)
            
            # Forward (no teacher forcing)
            outputs, attentions = model(src, src_len, trg, teacher_forcing_ratio=0)
            
            # Reshape
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:, :].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(outputs, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# ============================================
# VISUALIZE ATTENTION (Bonus)
# ============================================

def visualize_attention(src_sentence, trg_sentence, attention_weights, src_vocab, tgt_vocab):
    """
    Visualize attention heatmap
    Requires: matplotlib, seaborn
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert indices to words
    src_words = [src_vocab.idx2word[idx] for idx in src_sentence]
    trg_words = [tgt_vocab.idx2word[idx] for idx in trg_sentence]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights[:len(trg_words), :len(src_words)],
        xticklabels=src_words,
        yticklabels=trg_words,
        cmap='Blues',
        cbar=True
    )
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=150)
    plt.show()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("  TRAINING SEQ2SEQ WITH ATTENTION")
    print("  Target: BLEU ~45%")
    print("="*60)
    
    # 1. Load vocabularies (from your existing code)
    print("\n[1/5] Loading vocabularies...")
    # src_vocab = torch.load('check_point/src_vocab.pth')
    # tgt_vocab = torch.load('check_point/tgt_vocab.pth')
    
    # 2. Load data iterators (from your existing code)
    print("[2/5] Loading data iterators...")
    # train_iterator, val_iterator, test_iterator = create_iterators(...)
    
    # 3. Build model
    print("[3/5] Building model with Attention...")
    # model = build_attention_model(src_vocab, tgt_vocab, CONFIG)
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Train
    print("[4/5] Training...")
    # model = train_with_scheduler(model, train_iterator, val_iterator, CONFIG)
    
    # 5. Evaluate
    print("[5/5] Evaluating on test set...")
    # bleu_score = calculate_bleu_on_test_set(test_iterator, src_vocab, tgt_vocab, model, CONFIG['device'])
    # print(f"\n🎯 Final BLEU Score: {bleu_score:.2f}%")
    
    print("\n✅ Done!")
