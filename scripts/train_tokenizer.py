import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import train_sentencepiece_tokenizers
from src.config import Config

if __name__ == "__main__":
    train_sentencepiece_tokenizers(
        en_corpus=Config.TRAIN_EN_FILE,
        zh_corpus=Config.TRAIN_ZH_FILE,
        output_dir=Config.VOCAB_DIR,  # 需在 config.py 中添加
        vocab_size_en=Config.MAX_VOCAB,
        vocab_size_zh=Config.MAX_VOCAB_ZH
    )