# dataset.py - 全新版本

"""数据集模块（适配 SentencePiece）"""
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.tokenizer import SentencePieceTokenizer
from src.config import Config

class ParallelDataset(Dataset):
    def __init__(self, en_file, zh_file, en_tokenizer, zh_tokenizer, max_length=50):
        self.data = []
        self.en_tok = en_tokenizer
        self.zh_tok = zh_tokenizer
        self.max_length = max_length
        
        print(f"📖 加载数据集: {en_file}")
        
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(zh_file, 'r', encoding='utf-8') as f_zh:
            en_lines = sum(1 for _ in f_en)
            zh_lines = sum(1 for _ in f_zh)
        
        total = min(en_lines, zh_lines)
        skipped = 0
        
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(zh_file, 'r', encoding='utf-8') as f_zh:
            
            for en_line, zh_line in tqdm(zip(f_en, f_zh), total=total, desc="编码"):
                en_text = en_line.strip()
                zh_text = zh_line.strip()
                
                if not en_text or not zh_text:
                    skipped += 1
                    continue
                
                try:
                    # 先获取原始编码（不加 BOS/EOS）
                    en_ids_raw = self.en_tok.encode(en_text, out_type=int)
                    zh_ids_raw = self.zh_tok.encode(zh_text, out_type=int)
                    
                    # 手动添加 BOS (1) 和 EOS (0)
                    en_ids = [1] + en_ids_raw + [0]
                    zh_ids = [1] + zh_ids_raw + [0]
                    
                    # 调试打印前两条
                    if len(self.data) < 2:
                        print("EN:", en_text, "->", en_ids)
                        print("ZH:", zh_text, "->", zh_ids)
                        
                except Exception as e:
                    print(f"❌ Encode error on line {len(self.data)}: {e}")
                    skipped += 1
                    continue
                
                if len(en_ids) > max_length or len(zh_ids) > max_length:
                    skipped += 1
                    continue
                
                # OOV 检查（可选）
                en_unk_ratio = en_ids.count(self.en_tok.unk_id) / len(en_ids)
                zh_unk_ratio = zh_ids.count(self.zh_tok.unk_id) / len(zh_ids)
                if en_unk_ratio > 0.3 or zh_unk_ratio > 0.3:
                    skipped += 1
                    continue
                if en_unk_ratio > 0.5 or zh_unk_ratio > 0.5:  # 改为 50%
                    skipped += 1
                    continue         
                       
                self.data.append({
                    'src': torch.tensor(en_ids),
                    'tgt': torch.tensor(zh_ids),
                    'src_text': en_text,
                    'tgt_text': zh_text,
                })
        
        print(f"📊 数据集统计: 有效={len(self.data)}, 跳过={skipped}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """批处理：仅填充，mask 在训练时动态生成"""
    srcs = [item['src'] for item in batch]
    tgts = [item['tgt'] for item in batch]
    
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=3)  # PAD_ID=3
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=3)
    
    return {'src': src, 'tgt': tgt}

def create_dataloaders(config):    
    en_tok = SentencePieceTokenizer(f"{Config.VOCAB_DIR}/en.model")
    zh_tok = SentencePieceTokenizer(f"{Config.VOCAB_DIR}/ch.model")
    
    train_dataset = ParallelDataset(
        Config.TRAIN_EN_FILE,
        Config.TRAIN_ZH_FILE,
        en_tok, zh_tok,
        max_length=Config.MAX_LENGTH
    )
    
    val_dataset = ParallelDataset(
        Config.VAL_EN_FILE,
        Config.VAL_ZH_FILE,
        en_tok, zh_tok,
        max_length=Config.MAX_LENGTH
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("数据集为空！")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"📊 数据加载器: 训练={len(train_loader)}批, 验证={len(val_loader)}批")
    
    return train_loader, val_loader, en_tok, zh_tok