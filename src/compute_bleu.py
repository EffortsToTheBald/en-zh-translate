import sacrebleu

# 在 main() 最后添加：
def compute_bleu():
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