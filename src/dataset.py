"""数据集模块"""
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ParallelDataset(Dataset):
    """平行文本数据集"""
    
    def __init__(self, en_file, zh_file, en_vocab, zh_vocab, max_length=50):
        self.data = []
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        self.max_length = max_length
        
        print(f"📖 加载数据集: {en_file}")
        
        # 统计行数
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(zh_file, 'r', encoding='utf-8') as f_zh:
            
            en_lines = sum(1 for _ in f_en)
            zh_lines = sum(1 for _ in f_zh)
        
        lines = min(en_lines, zh_lines)
        print(f"  总行数: {lines}")
        
        # 读取和编码
        skipped = 0
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(zh_file, 'r', encoding='utf-8') as f_zh:
            
            for i, (en_line, zh_line) in enumerate(tqdm(zip(f_en, f_zh), total=lines, desc="编码")):
                en_text = en_line.strip()
                zh_text = zh_line.strip()
                
                if not en_text or not zh_text:
                    skipped += 1
                    continue
                
                # 编码
                en_indices = en_vocab.encode(en_text, add_special_tokens=True)
                zh_indices = zh_vocab.encode(zh_text, add_special_tokens=True)
                
                # 检查长度
                if (len(en_indices) <= max_length and 
                    len(zh_indices) <= max_length):
                    
                    # 检查未知词比例
                    unk_idx = en_vocab.word2idx[en_vocab.UNK_TOKEN]
                    en_unk_ratio = en_indices.count(unk_idx) / len(en_indices)
                    zh_unk_ratio = zh_indices.count(unk_idx) / len(zh_indices)
                    
                    if en_unk_ratio < 0.3 and zh_unk_ratio < 0.3:
                        self.data.append({
                            'src': torch.tensor(en_indices),
                            'tgt': torch.tensor(zh_indices),
                            'src_text': en_text,
                            'tgt_text': zh_text,
                            'src_len': len(en_indices),
                            'tgt_len': len(zh_indices)
                        })
                    else:
                        skipped += 1
                else:
                    skipped += 1
        
        print(f"📊 数据集统计:")
        print(f"  有效样本: {len(self.data)}")
        print(f"  跳过样本: {skipped}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """批处理函数 - 仅返回填充后的序列，mask 在训练时动态生成"""
    srcs = [item['src'] for item in batch]
    tgts = [item['tgt'] for item in batch]
    
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    
    return {
        'src': src,
        'tgt': tgt
    }
    # """批处理函数"""
    # srcs = [item['src'] for item in batch]
    # tgts = [item['tgt'] for item in batch]
    
    # src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    # tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    
    # src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    # tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    
    # seq_len = tgt.size(1)
    # nopeak_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    # tgt_mask = tgt_mask & nopeak_mask
    
    # return {
    #     'src': src,
    #     'tgt': tgt,
    #     'src_mask': src_mask,
    #     'tgt_mask': tgt_mask
    # }

def create_dataloaders(en_vocab, zh_vocab, config):
    """创建数据加载器"""
    from config import Config
    
    # 创建数据集
    train_dataset = ParallelDataset(
        Config.TRAIN_EN_FILE,
        Config.TRAIN_ZH_FILE,
        en_vocab, zh_vocab,
        max_length=Config.MAX_LENGTH
    )
    
    val_dataset = ParallelDataset(
        Config.VAL_EN_FILE,
        Config.VAL_ZH_FILE,
        en_vocab, zh_vocab,
        max_length=Config.MAX_LENGTH
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("数据集为空！")
    
    # 创建数据加载器
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
    
    print(f"📊 数据加载器统计:")
    print(f"  训练批次: {len(train_loader)} (批次大小: {Config.BATCH_SIZE})")
    print(f"  验证批次: {len(val_loader)} (批次大小: {Config.BATCH_SIZE})")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    
    return train_loader, val_loader