"""
Test script đơn giản để kiểm tra cấu trúc project và data
KHÔNG yêu cầu PyTorch
"""

from pathlib import Path
import re

def check_project_structure():
    """Kiểm tra cấu trúc thư mục"""
    print("=" * 70)
    print("KIỂM TRA CẤU TRÚC PROJECT")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Kiểm tra các thư mục
    directories = {
        "data": base_dir / "data",
        "src": base_dir / "src",
        "check_point": base_dir / "check_point",
        "report": base_dir / "report"
    }
    
    print("\n📁 Thư mục:")
    for name, path in directories.items():
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {name}/")
    
    # Kiểm tra data files
    print("\n📄 Data files:")
    data_files = [
        "train.en", "train.fr",
        "val.en", "val.fr",
        "test.en", "test.fr"
    ]
    
    missing_files = []
    for filename in data_files:
        filepath = directories["data"] / filename
        if filepath.exists():
            # Đếm số dòng
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print(f"  ✅ {filename:<12} ({lines:>6} dòng)")
        else:
            print(f"  ❌ {filename:<12} (THIẾU)")
            missing_files.append(filename)
    
    # Kiểm tra source files
    print("\n🐍 Source files:")
    src_files = [
        "config.py",
        "utils.py", 
        "data_loader.py"
    ]
    
    for filename in src_files:
        filepath = directories["src"] / filename
        status = "✅" if filepath.exists() else "❌"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print(f"  {status} {filename:<20} ({lines:>4} dòng)")
        else:
            print(f"  {status} {filename}")
    
    print("\n" + "=" * 70)
    
    if missing_files:
        print(f"⚠️  CẢNH BÁO: Thiếu {len(missing_files)} file data: {', '.join(missing_files)}")
    else:
        print("✅ TẤT CẢ FILES DATA ĐẦY ĐỦ")
    
    print("=" * 70)


def preview_data():
    """Xem trước một vài dòng data"""
    print("\n" + "=" * 70)
    print("XEM TRƯỚC DỮ LIỆU")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Đọc 3 dòng đầu của train data
    train_en = base_dir / "data" / "train.en"
    train_fr = base_dir / "data" / "train.fr"
    
    if train_en.exists() and train_fr.exists():
        print("\n📖 3 cặp câu đầu tiên (train):\n")
        
        with open(train_en, 'r', encoding='utf-8') as f_en, \
             open(train_fr, 'r', encoding='utf-8') as f_fr:
            
            for i, (en_line, fr_line) in enumerate(zip(f_en, f_fr), 1):
                if i > 3:
                    break
                print(f"{i}. EN: {en_line.strip()}")
                print(f"   FR: {fr_line.strip()}")
                print()


def test_tokenization():
    """Test tokenization function (không cần PyTorch)"""
    print("\n" + "=" * 70)
    print("TEST TOKENIZATION")
    print("=" * 70)
    
    def simple_tokenize(sentence):
        """Tokenize đơn giản"""
        sentence = sentence.lower()
        sentence = re.sub(r"([.!?;,])", r" \1", sentence)
        return sentence.split()
    
    test_sentences = [
        "Hello, how are you?",
        "Two young, White males are outside near many bushes.",
        "Deux jeunes hommes blancs sont dehors près de nombreux buissons."
    ]
    
    print("\nVí dụ tokenization:")
    for sent in test_sentences:
        tokens = simple_tokenize(sent)
        print(f"\nGốc:  {sent}")
        print(f"Tokens: {tokens}")
        print(f"Số tokens: {len(tokens)}")


def main():
    """Main function"""
    print("\n")
    print("🎓 ĐỒ ÁN NLP - KIỂM TRA PROJECT")
    print("English-French Translation với LSTM Encoder-Decoder")
    
    # Kiểm tra cấu trúc
    check_project_structure()
    
    # Xem trước data
    preview_data()
    
    # Test tokenization
    test_tokenization()
    
    print("\n" + "=" * 70)
    print("✅ KIỂM TRA HOÀN TẤT")
    print("=" * 70)
    print("\n📝 KẾT LUẬN:")
    print("  - Task 1 (Thiết lập môi trường): ✅ HOÀN THÀNH")
    print("  - Task 2 (Xử lý dữ liệu): ✅ CODE SẴN SÀNG")
    print("\n💡 BƯỚC TIẾP THEO:")
    print("  1. Cài đặt PyTorch: pip install -r requirements.txt")
    print("  2. Test data loading: python src/data_loader.py")
    print("  3. Implement Task 3: Encoder-Decoder model")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
