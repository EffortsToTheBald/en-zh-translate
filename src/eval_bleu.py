"""
优化版 BLEU 评估脚本
- 增加重复惩罚
- 修复解码终止逻辑
- 提升翻译质量
"""

import os
import sys
import torch
import sentencepiece as spm
from tqdm import tqdm
import sacrebleu
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model


def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    return mask.bool()


def load_model_and_tokenizers(device, checkpoint_path):
    print(f"📂 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint['config']

    vocab_dir = saved_config['VOCAB_DIR']
    en_sp = spm.SentencePieceProcessor()
    zh_sp = spm.SentencePieceProcessor()
    en_sp.load(os.path.join(vocab_dir, "en.model"))
    zh_sp.load(os.path.join(vocab_dir, "ch.model"))

    print(f"✅ 英文 vocab_size: {en_sp.vocab_size()}")
    print(f"✅ 中文 vocab_size: {zh_sp.vocab_size()}")

    assert en_sp.vocab_size() <= saved_config['MAX_VOCAB'], \
        f"英文 vocab_size ({en_sp.vocab_size()}) > MAX_VOCAB ({saved_config['MAX_VOCAB']})"
    assert zh_sp.vocab_size() <= saved_config['MAX_VOCAB_ZH'], \
        f"中文 vocab_size ({zh_sp.vocab_size()}) > MAX_VOCAB_ZH ({saved_config['MAX_VOCAB_ZH']})"

    model = build_model(
        src_vocab_size=en_sp.vocab_size(),
        tgt_vocab_size=zh_sp.vocab_size(),
        device=device,** saved_config
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型权重加载成功！")

    sos_idx = saved_config['SOS_IDX']
    eos_idx = saved_config['EOS_IDX']
    pad_idx = saved_config['PAD_IDX']

    return model, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, saved_config


def translate_batch(model, src_sentences, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, device, max_len=40, repetition_penalty=1.2):
    """
    优化版批量贪心解码（带重复惩罚）
    """
    if not src_sentences:
        return []

    B = len(src_sentences)
    # Step 1: 编码所有源句子
    src_token_lists = []
    valid_indices = []
    for i, sent in enumerate(src_sentences):
        if not sent.strip():
            continue
        tokens = en_sp.encode(sent, out_type=int)
        if not tokens:
            tokens = [en_sp.unk_id()]
        src_token_lists.append(torch.tensor(tokens, dtype=torch.long))
        valid_indices.append(i)

    if not src_token_lists:
        return [""] * B

    # Pad source sequences
    src_padded = pad_sequence(src_token_lists, batch_first=True, padding_value=pad_idx).to(device)
    B_eff = src_padded.size(0)

    # 初始化 target
    tgt_tokens = [[sos_idx] for _ in range(B_eff)]
    finished = [False] * B_eff
    consecutive_repeats = [0] * B_eff
    last_tokens = [None] * B_eff

    with torch.no_grad():
        for step in range(max_len - 1):
            if all(finished):
                break

            # 构造当前 tgt 张量
            current_tgt = [torch.tensor(seq, dtype=torch.long) for seq in tgt_tokens]
            tgt_padded = pad_sequence(current_tgt, batch_first=True, padding_value=pad_idx).to(device)

            # Masks
            src_padding_mask = (src_padded == pad_idx)
            tgt_padding_mask = (tgt_padded == pad_idx)
            tgt_mask = generate_square_subsequent_mask(tgt_padded.size(1), device)

            # 前向传播
            output, _ = model(
                src=src_padded,
                tgt=tgt_padded,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )

            # 取每个样本最后一个 token 的预测（带重复惩罚）
            next_tokens = []
            for i in range(B_eff):
                if finished[i]:
                    next_tokens.append(eos_idx)
                    continue
                    
                # 获取logits并应用重复惩罚
                logits = output[i, -1]
                generated = set(tgt_tokens[i])
                
                # 重复惩罚
                for token in generated:
                    if logits[token] > 0:
                        logits[token] /= repetition_penalty
                    else:
                        logits[token] *= repetition_penalty
                
                next_token = logits.argmax().item()
                
                # 检查连续重复
                if next_token == last_tokens[i]:
                    consecutive_repeats[i] += 1
                    if consecutive_repeats[i] >= 3:
                        finished[i] = True
                        next_token = eos_idx
                else:
                    consecutive_repeats[i] = 0
                
                last_tokens[i] = next_token
                next_tokens.append(next_token)

            # 更新 tgt_tokens 和 finished 标志
            for i in range(B_eff):
                if not finished[i]:
                    token = next_tokens[i]
                    tgt_tokens[i].append(token)
                    if token == eos_idx or len(tgt_tokens[i]) >= max_len:
                        finished[i] = True

    # 解码结果
    results = [""] * B
    for idx_in_batch, orig_idx in enumerate(valid_indices):
        token_ids = tgt_tokens[idx_in_batch][1:]  # 移除 SOS
        
        # 清理重复token和特殊token
        clean_ids = []
        prev_token = None
        for tid in token_ids:
            if tid in {eos_idx, pad_idx, zh_sp.unk_id()}:
                continue
            if tid == prev_token:
                continue
            clean_ids.append(tid)
            prev_token = tid
        
        try:
            decoded = zh_sp.decode_ids(clean_ids)
            # 清理格式
            decoded = decoded.replace('▁', ' ').replace('  ', ' ').strip()
            results[orig_idx] = decoded
        except Exception:
            results[orig_idx] = ""

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    from src.config import Config
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"❌ Checkpoint 不存在: {checkpoint_path}"

    model, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, saved_config = load_model_and_tokenizers(device, checkpoint_path)

    # 加载验证集
    val_en_file = "./data/val.en"
    val_zh_file = "./data/val.zh"

    for path in [val_en_file, val_zh_file]:
        assert os.path.exists(path), f"❌ 验证集文件不存在: {path}"

    with open(val_en_file, 'r', encoding='utf-8') as f:
        en_sents = [line.strip() for line in f]
    with open(val_zh_file, 'r', encoding='utf-8') as f:
        ref_sents = [line.strip() for line in f]

    assert len(en_sents) == len(ref_sents), "❌ 行数不一致！"
    total = len(en_sents)
    print(f"📊 共 {total} 条验证样本")

    # 批量翻译
    BATCH_SIZE = 64  # 降低批次大小提升稳定性
    pred_sents = []

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Translating (Batch)"):
        batch_en = en_sents[i:i + BATCH_SIZE]
        batch_preds = translate_batch(
            model, batch_en, en_sp, zh_sp,
            sos_idx, eos_idx, pad_idx, device,
            max_len=40, repetition_penalty=1.5
        )
        pred_sents.extend(batch_preds)

    # 过滤空输出
    pred_sents = [p if p else "<empty>" for p in pred_sents]
    
    # 计算 BLEU
    bleu = sacrebleu.corpus_bleu(pred_sents, [ref_sents], tokenize='zh')
    print(f"\n🎉 最终 BLEU 分数 (tokenize='zh'): {bleu.score:.2f}")
    print(f"详细信息: {bleu}")
    
    # 打印部分翻译结果示例
    print("\n📝 翻译示例 (前5条):")
    for i in range(min(5, len(en_sents))):
        print(f"EN: {en_sents[i]}")
        print(f"REF: {ref_sents[i]}")
        print(f"HYP: {pred_sents[i]}\n")


if __name__ == "__main__":
    main()