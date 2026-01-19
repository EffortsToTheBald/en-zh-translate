"""
集束搜索评估脚本
"""
import os
import sys
import torch
import sentencepiece as spm
from tqdm import tqdm
import sacrebleu
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    return mask.bool()

def load_model_and_tokenizers(device, checkpoint_path):
    """与原有相同的加载函数"""
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
        device=device,
        **saved_config
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    sos_idx = saved_config['SOS_IDX']
    eos_idx = saved_config['EOS_IDX']
    pad_idx = saved_config['PAD_IDX']
    
    return model, en_sp, zh_sp, sos_idx, eos_idx, pad_idx

def beam_search_translate(model, src_sentences, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, device,
                         beam_size=5, max_len=60, length_penalty=0.6):
    """集束搜索翻译"""
    translations = []
    
    for src_text in tqdm(src_sentences, desc="Beam Search"):
        if not src_text.strip():
            translations.append("")
            continue
        
        # 编码
        src_tokens = en_sp.encode(src_text, out_type=int)
        if not src_tokens:
            translations.append("")
            continue
        
        src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # 编码器
        src_emb = model.src_embedding(src_tensor) * math.sqrt(model.d_model)
        src_emb = model.positional_encoding(src_emb)
        memory = model.transformer.encoder(src_emb)
        
        # 初始化束
        beams = [{'tokens': [sos_idx], 'score': 0.0, 'hidden': None}]
        finished = []
        
        for step in range(max_len):
            new_beams = []
            
            for beam in beams:
                if beam['tokens'][-1] == eos_idx:
                    finished.append(beam)
                    continue
                
                # 准备解码器输入
                tgt_tensor = torch.tensor(beam['tokens'], dtype=torch.long).unsqueeze(0).to(device)
                tgt_emb = model.tgt_embedding(tgt_tensor) * math.sqrt(model.d_model)
                tgt_emb = model.positional_encoding(tgt_emb)
                
                # 解码
                tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1), device)
                output = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)[0]
                
                # top-k候选
                topk_probs, topk_indices = torch.topk(log_probs, beam_size)
                
                for j in range(beam_size):
                    new_tokens = beam['tokens'] + [topk_indices[j].item()]
                    new_score = beam['score'] + topk_probs[j].item()
                    new_beams.append({
                        'tokens': new_tokens,
                        'score': new_score,
                        'hidden': None
                    })
            
            # 选择top beam_size
            beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_size]
            
            # 检查结束条件
            if all(b['tokens'][-1] == eos_idx for b in beams):
                finished.extend(beams)
                break
        
        # 如果没有完成的，使用当前最佳
        if not finished:
            finished = beams
        
        # 选择最佳（带长度归一化）
        best_beam = None
        best_score = -float('inf')
        
        for beam in finished:
            length = len(beam['tokens'])
            if length_penalty != 1.0:
                score = beam['score'] / (length ** length_penalty)
            else:
                score = beam['score'] / length
            
            if score > best_score:
                best_score = score
                best_beam = beam
        
        # 解码
        if best_beam:
            clean_ids = [
                tid for tid in best_beam['tokens'] 
                if tid not in {sos_idx, eos_idx, pad_idx, zh_sp.unk_id()}
            ]
            try:
                translation = zh_sp.decode_ids(clean_ids).strip()
                # 清理特殊字符
                translation = translation.replace("<s>", "").replace("</s>", "").replace("<pad>", "").replace("<unk>", "")
                translations.append(translation)
            except:
                translations.append("")
        else:
            translations.append("")
    
    return translations

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    from src.config import Config
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    
    # 加载模型
    model, en_sp, zh_sp, sos_idx, eos_idx, pad_idx = load_model_and_tokenizers(device, checkpoint_path)
    
    # 测试句子
    test_sentences = [
        "Hello world.",
        "I love you.",
        "What is your name?",
        "The weather is nice today.",
        "Good morning!",
        "Thank you very much.",
        "How are you doing?",
        "What time is it?",
        "I need your help.",
        "This is a beautiful place."
    ]
    
    # 对比不同解码方法
    print("\n🔍 对比不同解码方法:")
    
    # 贪心解码（原方法）
    from eval_bleu import translate_batch
    greedy_results = translate_batch(
        model, test_sentences, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, device
    )
    
    # 集束搜索（beam_size=3）
    beam3_results = beam_search_translate(
        model, test_sentences, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, device,
        beam_size=3, max_len=50, length_penalty=0.6
    )
    
    # 集束搜索（beam_size=5）
    beam5_results = beam_search_translate(
        model, test_sentences, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, device,
        beam_size=5, max_len=50, length_penalty=0.6
    )
    
    # 打印结果对比
    print("\n📊 翻译结果对比:")
    for i, (en, greedy, beam3, beam5) in enumerate(zip(test_sentences, greedy_results, beam3_results, beam5_results)):
        print(f"\n{i+1}. EN: {en}")
        print(f"   贪心: {greedy}")
        print(f"   束3:  {beam3}")
        print(f"   束5:  {beam5}")
    
    # 评估验证集
    print("\n📈 评估验证集上的表现...")
    
    val_en_file = "./data/val.en"
    val_zh_file = "./data/val.zh"
    
    with open(val_en_file, 'r', encoding='utf-8') as f:
        en_sents = [line.strip() for line in f][:100]  # 先用100条测试
    
    with open(val_zh_file, 'r', encoding='utf-8') as f:
        ref_sents = [line.strip() for line in f][:100]
    
    # 集束搜索翻译
    beam_translations = beam_search_translate(
        model, en_sents, en_sp, zh_sp, sos_idx, eos_idx, pad_idx, device,
        beam_size=5, max_len=60, length_penalty=0.7
    )
    
    # 计算BLEU
    bleu = sacrebleu.corpus_bleu(beam_translations, [ref_sents], tokenize='zh')
    print(f"\n🎉 集束搜索 (beam=5) BLEU 分数: {bleu.score:.2f}")
    print(f"详细信息: {bleu}")

if __name__ == "__main__":
    main()