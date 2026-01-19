"""推理模块"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from config import Config
from vocabulary import Vocabulary
from model import build_model

# 工具函数：生成下三角掩码（用于解码）
def generate_square_subsequent_mask(sz):
    """生成 [sz, sz] 的因果掩码"""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    return mask == 1  # True 表示要屏蔽的位置

def load_model(model_path):
    """加载模型"""
    print(f"加载模型: {model_path}")
    
    # 加载词汇表
    en_vocab = Vocabulary.load(f"{Config.VOCAB_DIR}/en_vocab.pkl")
    zh_vocab = Vocabulary.load(f"{Config.VOCAB_DIR}/zh_vocab.pkl")
    
    # 构建模型（注意：build_model 内部已 .to(device)）
    device = Config.DEVICE
    model = build_model(len(en_vocab), len(zh_vocab), device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"训练轮数: {checkpoint['epoch']}")
    print(f"验证损失: {checkpoint['val_loss']:.4f}")
    
    return model, en_vocab, zh_vocab, device

def translate(model, sentence, en_vocab, zh_vocab, device, temperature=0.8, max_len=50):
    """翻译单句（greedy + temperature sampling）"""
    model.eval()
    
    # 1. 编码输入句子（假设 Vocabulary.encode 返回带 <sos>/<eos> 的 ID 列表）
    src_indices = en_vocab.encode(sentence, add_special_tokens=True)
    src = torch.tensor(src_indices).unsqueeze(0).to(device)  # [1, S]
    
    # 2. 准备目标序列起始
    sos_id = zh_vocab.word2idx[Config.SOS_TOKEN]
    eos_id = zh_vocab.word2idx[Config.EOS_TOKEN]
    pad_id = zh_vocab.word2idx[Config.PAD_TOKEN]
    
    tgt_indices = [sos_id]  # 起始 token
    
    with torch.no_grad():
        for i in range(max_len - 1):  # 预留 <eos>
            tgt = torch.tensor(tgt_indices).unsqueeze(0).to(device)  # [1, T]
            
            # 构造 masks
            src_padding_mask = (src == en_vocab.word2idx[Config.PAD_TOKEN])       # [1, S]
            tgt_padding_mask = (tgt == pad_id)                                   # [1, T]
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)   # [T, T]
            
            # 前向传播
            output, _ = model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )  # output: [1, T, vocab_size]
            
            # 取最后一个 token 的 logits
            next_token_logits = output[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            tgt_indices.append(next_token)
            
            if next_token == eos_id:
                break
        
        # 解码（跳过 <sos>，遇到 <eos> 停止）
        translation = zh_vocab.decode(tgt_indices[1:])  # decode 内部应处理 <eos>
        return translation

def main():
    """主推理函数"""
    print("🔤 翻译测试")
    print("=" * 60)
    
    model, en_vocab, zh_vocab, device = load_model(f"{Config.CHECKPOINT_DIR}/best_model.pth")
    
    test_sentences = [
        "A group of men are loading cotton onto a truck",
        "A man sleeping in a green room on a couch.",
        "A boy wearing headphones sits on a woman's shoulders.",
        "Two people are building a blue ice house by the lake",
        "A woman is cooking food in the kitchen",
        "A dog is running in the park",
        "A cat is sleeping on the sofa",
        "Children are playing in the playground",
        "Nice to meet you",
        "Hello world"
    ]
    
    print("\n📝 翻译结果:")
    for sentence in test_sentences:
        try:
            translation = translate(model, sentence, en_vocab, zh_vocab, device, temperature=0.8)
            print(f"英文: {sentence}")
            print(f"中文: {translation}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ 翻译失败: {e}")
            print("-" * 40)

if __name__ == "__main__":
    main()