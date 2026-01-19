"""工具函数"""
import os
import sys
import torch
import torch.nn as nn
from src.config import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose  # 新增
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"✅ 验证损失下降 ({self.best_loss:.6f} → {val_loss:.6f})，重置早停计数器。")
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ 验证损失未改善，早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("🛑 触发早停！")
        return self.early_stop
    
class LabelSmoothingLoss(nn.Module):
    """标签平滑损失（正确实现）"""
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        # 创建非 PAD 掩码
        non_pad_mask = (target != self.padding_idx)
        
        # 计算 NLL Loss（自动忽略 PAD）
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        nll_loss = nll_loss.masked_select(non_pad_mask).mean()
        
        # 平滑损失：对所有非 PAD 位置均匀分布
        smooth_loss = -log_probs.mean(dim=-1)
        smooth_loss = smooth_loss.masked_select(non_pad_mask).mean()
        
        return self.confidence * nll_loss + self.smoothing * smooth_loss

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    return mask.bool()

class BeamSearchNode:
    """Beam Search节点"""
    def __init__(self, hidden_state, previous_node, token_id, log_prob, length):
        self.hidden_state = hidden_state
        self.prev_node = previous_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        """计算节点分数（带长度惩罚）"""
        reward = 0
        # 长度惩罚：鼓励更短的序列
        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward

def beam_search_decode(
    model, src, src_padding_mask, sos_idx, eos_idx, pad_idx, 
    device, max_len=60, beam_size=4, repetition_penalty=1.8, temperature=0.95
):
    """
    Beam Search解码
    """
    src_seq_len = src.size(1)
    batch_size = src.size(0)
    
    # 初始化beam
    nodes = []
    for b in range(batch_size):
        node = BeamSearchNode(None, None, sos_idx, 0.0, 1)
        nodes.append([node])
    
    # 存储最终结果
    final_outputs = [[] for _ in range(batch_size)]
    
    for step in range(max_len):
        # 收集当前所有候选token
        current_tokens = []
        current_hidden = []
        current_nodes = []
        
        for b in range(batch_size):
            if len(final_outputs[b]) > 0:
                continue
                
            beam_nodes = nodes[b]
            for node in beam_nodes:
                current_tokens.append(node.token_id)
                current_nodes.append(node)
        
        if not current_tokens:
            break
        
        # 构建输入
        tgt_input = torch.tensor([[n.token_id for n in nodes[b]] for b in range(batch_size)], 
                                dtype=torch.long, device=device)
        
        # 处理mask
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
        tgt_padding_mask = (tgt_input == pad_idx)
        
        # 前向传播
        with torch.no_grad():
            output, _ = model(
                src=src,
                tgt=tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
        
        # 处理每个batch
        for b in range(batch_size):
            if len(final_outputs[b]) > 0:
                continue
                
            beam_nodes = nodes[b]
            new_nodes = []
            
            for i, node in enumerate(beam_nodes):
                # 获取当前步的logits
                logits = output[b, i, :] / temperature
                
                # 重复惩罚
                generated_tokens = []
                curr_node = node
                while curr_node.prev_node is not None:
                    generated_tokens.append(curr_node.token_id)
                    curr_node = curr_node.prev_node
                
                for token in generated_tokens:
                    if logits[token] > 0:
                        logits[token] /= repetition_penalty
                    else:
                        logits[token] *= repetition_penalty
                
                # 计算概率
                probs = F.log_softmax(logits, dim=-1)
                
                # 取top-k
                top_probs, top_tokens = probs.topk(beam_size)
                
                for j in range(beam_size):
                    token = top_tokens[j].item()
                    prob = top_probs[j].item()
                    
                    new_node = BeamSearchNode(
                        None, node, token, node.log_prob + prob, node.length + 1
                    )
                    new_nodes.append(new_node)
            
            # 筛选最佳节点
            new_nodes = sorted(new_nodes, key=lambda x: x.eval(), reverse=True)[:beam_size]
            
            # 检查是否完成
            complete = False
            for node in new_nodes:
                if node.token_id == eos_idx:
                    # 回溯获取完整序列
                    sequence = []
                    curr_node = node
                    while curr_node.prev_node is not None:
                        sequence.append(curr_node.token_id)
                        curr_node = curr_node.prev_node
                    sequence.reverse()
                    final_outputs[b] = sequence
                    complete = True
                    break
            
            if not complete:
                nodes[b] = new_nodes
    
    # 处理未完成的序列
    for b in range(batch_size):
        if len(final_outputs[b]) == 0:
            # 取得分最高的节点
            best_node = sorted(nodes[b], key=lambda x: x.eval(), reverse=True)[0]
            sequence = []
            curr_node = best_node
            while curr_node.prev_node is not None:
                sequence.append(curr_node.token_id)
                curr_node = curr_node.prev_node
            sequence.reverse()
            final_outputs[b] = sequence
    
    return final_outputs