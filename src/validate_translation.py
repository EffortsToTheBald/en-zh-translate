"""使用验证集评估翻译模型"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import Config
from vocabulary import Vocabulary
from model import build_model
import torch.nn.functional as F
import sacrebleu
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    return mask == 1

def load_model(model_path, device):
    en_vocab = Vocabulary.load(f"{Config.VOCAB_DIR}/en_vocab.pkl")
    zh_vocab = Vocabulary.load(f"{Config.VOCAB_DIR}/zh_vocab.pkl")
    
    model = build_model(len(en_vocab), len(zh_vocab), device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, en_vocab, zh_vocab

def translate_sentence(model, sentence, en_vocab, zh_vocab, device, max_len=50):
    model.eval()
    src_indices = en_vocab.encode(sentence, add_special_tokens=True)
    src = torch.tensor(src_indices).unsqueeze(0).to(device)
    
    sos_id = zh_vocab.word2idx[Config.SOS_TOKEN]
    eos_id = zh_vocab.word2idx[Config.EOS_TOKEN]
    pad_id = zh_vocab.word2idx[Config.PAD_TOKEN]
    
    tgt_indices = [sos_id]
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            src_padding_mask = (src == en_vocab.word2idx[Config.PAD_TOKEN])
            tgt_padding_mask = (tgt == pad_id)
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            output, _ = model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )
            
            next_token = output[0, -1].argmax().item()  # greedy
            tgt_indices.append(next_token)
            if next_token == eos_id:
                break
        
        # decode 并移除 <eos>（假设 decode 能处理）
        return zh_vocab.decode(tgt_indices[1:])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, en_vocab, zh_vocab = load_model(f"{Config.CHECKPOINT_DIR}/best_model.pth", device)
    
    en_file = "./data/val.en"
    zh_file = "./data/val.zh"
    output_file = f"./{Config.RESULTS_DIR}/val_translations.txt"
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    with open(en_file, 'r', encoding='utf-8') as fen, \
         open(zh_file, 'r', encoding='utf-8') as fzh, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        en_lines = fen.readlines()
        zh_lines = fzh.readlines()
        
        assert len(en_lines) == len(zh_lines), "验证集英文和中文行数不一致！"
        
        print(f"共 {len(en_lines)} 条验证样本，开始翻译...")
        
        for i, (en_line, zh_line) in enumerate(zip(en_lines, zh_lines)):
            en_sent = en_line.strip()
            ref_sent = zh_line.strip()
            
            if not en_sent:
                pred_sent = ""
            else:
                try:
                    pred_sent = translate_sentence(model, en_sent, en_vocab, zh_vocab, device)
                except Exception as e:
                    print(f"第 {i+1} 行翻译出错: {e}")
                    pred_sent = "<ERROR>"
            
            # 写入格式：英文 ||| 参考译文 ||| 模型译文
            fout.write(f"{en_sent} ||| {ref_sent} ||| {pred_sent}\n")
            
            if (i + 1) % 50 == 0:
                print(f"已处理 {i+1} / {len(en_lines)}")
    
    print(f"✅ 翻译完成！结果已保存至: {output_file}")
    compute_bleu(output_file)

def compute_bleu(output_file):
    refs = []
    preds = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ||| ")
            if len(parts) == 3:
                _, ref, pred = parts
                refs.append(ref)
                preds.append(pred)
    
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    print(f"BLEU 分数: {bleu.score:.2f}")    

if __name__ == "__main__":
    main()