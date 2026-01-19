"""
修复版诊断脚本：解决重复生成 + 优化解码逻辑
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import sentencepiece as spm
from src.config import Config

from src.model import build_model


def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    return mask.bool()


def load_model_and_tokenizers(device, checkpoint_path):
    """精简版加载函数"""
    print(f"📂 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint['config']
    
    vocab_dir = saved_config['VOCAB_DIR']
    en_sp = spm.SentencePieceProcessor()
    zh_sp = spm.SentencePieceProcessor()
    en_sp.load(os.path.join(vocab_dir, "en.model"))
    zh_sp.load(os.path.join(vocab_dir, "ch.model"))

    model = build_model(
        src_vocab_size=en_sp.vocab_size(),
        tgt_vocab_size=zh_sp.vocab_size(),
        device=device,** saved_config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sos_idx = saved_config['SOS_IDX']
    eos_idx = saved_config['EOS_IDX']
    pad_idx = saved_config['PAD_IDX']

    return model, en_sp, zh_sp, sos_idx, eos_idx, pad_idx


def translate_sentence_debug(model, sentence, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, device, max_len=50, repetition_penalty=1.2):
    """带重复惩罚的优化解码函数"""
    print(f"\n--- 翻译输入: '{sentence}' ---")
    
    # 1. 编码源句
    src_tokens = en_sp.encode(sentence, out_type=int)
    print(f"  [EN] Tokens: {src_tokens}")
    print(f"  [EN] Decoded back: '{en_sp.decode_ids(src_tokens)}'")
    
    if not src_tokens:
        print("  ⚠️ 源句编码为空！")
        return "", []

    src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # 2. 解码过程（带重复惩罚）
    tgt_indices = [sos_idx]
    print(f"  [ZH] 初始 tgt: {tgt_indices} (SOS={sos_idx})")
    
    # 记录已生成的token，用于重复惩罚
    generated_tokens = set()
    consecutive_repeats = 0
    last_token = None

    with torch.no_grad():
        for step in range(max_len - 1):
            tgt = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
            src_padding_mask = (src == pad_idx)
            tgt_padding_mask = (tgt == pad_idx)
            tgt_mask = generate_square_subsequent_mask(tgt.size(1), device)

            output, _ = model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            # 获取最后一个token的logits
            logits = output[0, -1]
            
            # 重复惩罚：降低已生成token的概率
            for token in generated_tokens:
                if logits[token] > 0:
                    logits[token] /= repetition_penalty
                else:
                    logits[token] *= repetition_penalty
            
            # 预测下一个token
            next_token = logits.argmax().item()
            
            # 检查连续重复
            if next_token == last_token:
                consecutive_repeats += 1
                if consecutive_repeats >= 3:  # 连续3次生成相同token则终止
                    print(f"  ⚠️ 连续重复生成token {next_token}，强制终止")
                    break
            else:
                consecutive_repeats = 0
            
            last_token = next_token
            generated_tokens.add(next_token)
            tgt_indices.append(next_token)

            # 打印前5步
            if step < 5:
                piece = zh_sp.id_to_piece(next_token)
                print(f"    Step {step+1}: predicted ID={next_token}, piece='{piece}'")

            # 终止条件
            if next_token == eos_idx or len(tgt_indices) >= max_len:
                break

    print(f"  [ZH] 最终 IDs: {tgt_indices}")
    
    # 3. 安全解码
    clean_ids = []
    for tid in tgt_indices:
        if tid in {sos_idx, eos_idx, pad_idx, zh_sp.unk_id()}:
            continue
        # 过滤连续重复的token
        if clean_ids and clean_ids[-1] == tid:
            continue
        clean_ids.append(tid)
    
    print(f"  [ZH] Clean IDs: {clean_ids}")
    
    try:
        decoded = zh_sp.decode_ids(clean_ids).strip()
        # 清理多余空格
        decoded = decoded.replace('▁', ' ').replace('  ', ' ').strip()
        print(f"  [ZH] 最终译文: '{decoded}'")
        return decoded, tgt_indices
    except Exception as e:
        print(f"  ❌ 解码失败: {e}")
        return "", tgt_indices


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"❌ Checkpoint 不存在: {checkpoint_path}"

    # 加载
    model, en_sp, zh_sp, sos_idx, eos_idx, pad_idx = load_model_and_tokenizers(device, checkpoint_path)
    print("✅ 模型 transformer.batch_first =", model.transformer.batch_first)
    
    # 关键诊断
    print("\n🔍 Special Token 检查:")
    try:
        print(f"  SOS (ID={sos_idx}): '{en_sp.id_to_piece(sos_idx)}' (EN) | '{zh_sp.id_to_piece(sos_idx)}' (ZH)")
        print(f"  EOS (ID={eos_idx}): '{en_sp.id_to_piece(eos_idx)}' (EN) | '{zh_sp.id_to_piece(eos_idx)}' (ZH)")
        print(f"  PAD (ID={pad_idx}): '{en_sp.id_to_piece(pad_idx)}' (EN) | '{zh_sp.id_to_piece(pad_idx)}' (ZH)")
        print(f"  UNK (ID={en_sp.unk_id()}): '{en_sp.id_to_piece(en_sp.unk_id())}' (EN)")
        print(f"  UNK (ID={zh_sp.unk_id()}): '{zh_sp.id_to_piece(zh_sp.unk_id())}' (ZH)")
    except Exception as e:
        print(f"  ❌ 获取 special token 失败: {e}")

    # 测试翻译
    test_sentences = [
        "Hello world.",
        "I love you.",
        "What is your name?",
        "The weather is nice today.",
        "Good morning!"
    ]

    print("\n🧪 开始测试翻译...")
    results = []
    for sent in test_sentences:
        pred, ids = translate_sentence_debug(
            model, sent, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, 
            device, max_len=30, repetition_penalty=1.5  # 增加重复惩罚
        )
        results.append((sent, pred))
        if not pred or "<unk>" in pred:
            print("  ⚠️ 警告：检测到无效输出！")

    # 总结
    print("\n" + "="*50)
    print("📊 翻译结果汇总:")
    for en, zh in results:
        print(f"  EN: {en}")
        print(f"  ZH: {zh}\n")

    all_bad = all(not zh or "<unk>" in zh or len(zh) > 50 for _, zh in results)
    if all_bad:
        print("❌ 诊断结论: 模型输出异常！建议：")
        print("   1. 增加训练轮数（至少20轮）")
        print("   2. 降低学习率或调整批次大小")
        print("   3. 检查训练数据质量")
    else:
        print("✅ 诊断结论: 模型能输出有效中文，翻译质量可通过更多训练优化。")


if __name__ == "__main__":
    main()