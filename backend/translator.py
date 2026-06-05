# backend/translator.py
import os
import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.config import Config
from src.model import build_model


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """生成 float 类型的下三角 mask，-inf 表示屏蔽"""
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask


class EN2ZHTranslator:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
        
        print(f"🔄 正在加载模型: {model_path}")
        self.device = torch.device(Config.DEVICE)

        # --- 1. 加载 SentencePiece Tokenizers ---
        self.en_sp = spm.SentencePieceProcessor()
        self.zh_sp = spm.SentencePieceProcessor()
        en_model_path = os.path.join(Config.VAL_DIR, "en.model")
        zh_model_path = os.path.join(Config.VAL_DIR, "ch.model")
        self.en_sp.load(en_model_path)
        self.zh_sp.load(zh_model_path)

        # --- 2. ⚠️ 强制使用与训练时 dataset.py 一致的特殊 token ID ---
        # 在 dataset.py 中你写的是: [1] + ... + [0]，且 padding_value=3
        self.sos_id = 1   # BOS / SOS
        self.eos_id = 0   # EOS （注意：这通常是 <unk>，但你的模型把它当 EOS）
        self.pad_id = 3   # PAD

        # 可选：打印确认
        print(f"🔧 使用固定特殊 token ID: SOS={self.sos_id}, EOS={self.eos_id}, PAD={self.pad_id}")
        print(f"（注意：这覆盖了 SentencePiece 的默认值）")

        # --- 3. 加载模型 checkpoint ---
        checkpoint = torch.load(model_path, map_location=self.device)
        saved_config = checkpoint["config"]

        model_kwargs = {
            'd_model': saved_config.get('D_MODEL', 512),
            'nhead': saved_config.get('NHEAD', 8),
            'num_encoder_layers': saved_config.get('NUM_ENCODER_LAYERS', 6),
            'num_decoder_layers': saved_config.get('NUM_DECODER_LAYERS', 6),
            'dim_feedforward': saved_config.get('DIM_FEEDFORWARD', 2048),
            'dropout': saved_config.get('DROPOUT', 0.1),
        }

        self.model = build_model(
            src_vocab_size=self.en_sp.vocab_size(),
            tgt_vocab_size=self.zh_sp.vocab_size(),
            device=self.device,
            **model_kwargs
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval().to(self.device)
        print("✅ 模型加载成功！")

    def translate(self, sentence: str, temperature=0.8, max_len=50) -> str:
        self.model.eval()
        with torch.no_grad():
            src_pieces = self.en_sp.encode(sentence, out_type=int)
            src = torch.tensor(src_pieces).unsqueeze(0).to(self.device)

            tgt_indices = [self.sos_id]  # = [1]

            for _ in range(max_len - 1):
                tgt = torch.tensor(tgt_indices).unsqueeze(0).to(self.device)

                src_padding_mask = (src == self.pad_id).to(self.device)      # pad_id=3
                tgt_padding_mask = (tgt == self.pad_id).to(self.device)
                tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(self.device)

                output, _ = self.model(
                    src=src,
                    tgt=tgt,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                next_token_logits = output[0, -1, :] / temperature

                # 防御：处理 NaN/Inf
                if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                    next_token = self.eos_id  # 或 self.pad_id，但用 EOS 更合理（提前结束）
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    if torch.isnan(probs).any() or probs.sum() <= 1e-8:
                        next_token = self.eos_id
                    else:
                        next_token = torch.multinomial(probs, num_samples=1).item()

                # 🔒 强制约束 ID 范围 [0, vocab_size)
                if next_token < 0 or next_token >= self.zh_sp.vocab_size():
                    next_token = self.eos_id  # 安全 fallback

                tgt_indices.append(next_token)

                if next_token == self.eos_id:  # = 0
                    break

            # ✅ 关键修复：只取 SOS 之后、EOS 之前的 token
            generated_ids = []
            for tid in tgt_indices[1:]:  # 跳过开头的 SOS (1)
                if tid == self.eos_id:   # 遇到 EOS 停止，且不包含它
                    break
                generated_ids.append(tid)

            # 解码：跳过开头的 SOS (1)
            decoded = self.zh_sp.decode_ids(generated_ids)
            return decoded.strip()