"""
SentencePiece Tokenizer 封装
支持 BPE 分词，自动处理 <s>, </s>, <pad>, <unk>
"""

import sentencepiece as spm
import os
from typing import List, Union

class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def encode(self, text: str, out_type=str) -> Union[List[str], List[int]]:
        """编码文本为 tokens 或 IDs"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        return self.sp.encode(text, out_type=out_type)
    
    def decode(self, tokens: Union[List[str], List[int]]) -> str:
        """解码 tokens 或 IDs 为文本"""
        if not tokens:
            return ""
        if isinstance(tokens[0], int):
            return self.sp.decode_ids(tokens)
        else:
            return self.sp.decode_pieces(tokens)

    def id_to_piece(self, idx):
        return self.sp.id_to_piece(idx)

    def piece_to_id(self, piece):
        return self.sp.piece_to_id(piece)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()
    
    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()
    
    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()
    
    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()
    
    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

def train_sentencepiece_tokenizers(
    en_corpus: str,
    zh_corpus: str,
    output_dir: str,
    vocab_size_en: int = 32000,
    vocab_size_zh: int = 16000
):
    """
    训练英文和中文的 SentencePiece 模型
        en_corpus:  data/train.en
        zh_corpus:  data/train.zh
        output_dir: vocab_test
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练英文 (BPE)
    spm.SentencePieceTrainer.train(
        input=en_corpus,
        model_prefix=os.path.join(output_dir, "en_test"),
        vocab_size=vocab_size_en,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=3, unk_id=2, bos_id=1, eos_id=0,
        user_defined_symbols=["<pad>", "<s>", "</s>"]
    )
    # BPE
    # 从字符开始，贪心地合并最常见相邻对。
    # 合并后的单元成为新“字符”，参与下一轮统计。
    # 最终词表大小由你设定（如 32000），达到即停止。
    # 共享子结构提升泛化能力    

    # 训练中文 (BPE)
    spm.SentencePieceTrainer.train(
        input=zh_corpus,
        model_prefix=os.path.join(output_dir, "ch_test"),
        vocab_size=vocab_size_zh,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=3, unk_id=2, bos_id=1, eos_id=0,
        user_defined_symbols=["<pad>", "<s>", "</s>"]
    )
    
    print(f"✅ 英文 tokenizer 保存至: {output_dir}/en_test.model")
    print(f"✅ 中文 tokenizer 保存至: {output_dir}/ch_test.model")