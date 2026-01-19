"""训练模块"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import math
from datetime import datetime

from config import Config
from vocabulary import Vocabulary
from dataset import create_dataloaders
from model import build_model
from utils import EarlyStopping, LabelSmoothingLoss

# def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
#     """训练一个epoch"""
#     model.train()
#     total_loss = 0
#     total_tokens = 0
    
#     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    
#     for batch_idx, batch in enumerate(progress_bar):
#         src = batch['src'].to(device)
#         tgt = batch['tgt'].to(device)
#         src_mask = batch['src_mask'].to(device)
#         tgt_mask = batch['tgt_mask'].to(device)
        
#         # 准备输入输出
#         tgt_input = tgt[:, :-1]
#         tgt_output = tgt[:, 1:]
        
#         # 前向传播
#         optimizer.zero_grad()
#         output, _ = model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])
        
#         # 计算损失
#         loss = criterion(
#             output.contiguous().view(-1, output.size(-1)),
#             tgt_output.contiguous().view(-1)
#         )
        
#         # 反向传播
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)
#         optimizer.step()
        
#         # 统计
#         batch_tokens = (tgt_output != 0).sum().item()
#         total_loss += loss.item() * batch_tokens
#         total_tokens += batch_tokens
        
#         # 更新进度条
#         if batch_idx % 10 == 0:
#             progress_bar.set_postfix({
#                 'loss': loss.item(),
#                 'lr': optimizer.param_groups[0]['lr']
#             })
    
#     return total_loss / total_tokens if total_tokens > 0 else 0

# def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
#     model.train()
#     total_loss = 0
    
#     for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"训练 epoch {epoch}")):
#         src = batch['src'].to(device)  # [batch_size, src_len]
#         tgt = batch['tgt'].to(device)  # [batch_size, tgt_len]
        
#         # 创建目标输入（去掉最后一个token）
#         tgt_input = tgt[:, :-1]  # [batch_size, tgt_len-1]
        
#         # 创建填充掩码（2D张量）
#         src_padding_mask = (src == 0)  # [batch_size, src_len]
#         tgt_padding_mask = (tgt_input == 0)  # [batch_size, tgt_len-1]
        
#         # 创建因果掩码（防止看到未来信息）
#         tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)  # [tgt_len-1, tgt_len-1]
        
#         # 调用模型
#         output, _ = model(
#             src=src,
#             tgt=tgt_input,
#             tgt_mask=tgt_mask,
#             src_padding_mask=src_padding_mask,
#             tgt_padding_mask=tgt_padding_mask
#         )
        
#         # 计算损失
#         # output shape: [batch_size, tgt_len-1, vocab_size]
#         # target shape: [batch_size, tgt_len-1]
#         target = tgt[:, 1:]  # 去掉第一个token（<sos>）
        
#         output_flat = output.reshape(-1, output.size(-1))
#         target_flat = target.reshape(-1)
        
#         loss = criterion(output_flat, target_flat)
        
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         total_loss += loss.item()
        
#         # 可选：每N个批次打印一次损失
#         if batch_idx % 100 == 0:
#             print(f"批次 {batch_idx}, 损失: {loss.item():.4f}")
    
#     return total_loss / len(train_loader)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, pad_idx):
    model.train()
    total_loss = 0
    total_tokens = 0  # 新增
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"训练 epoch {epoch}")):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        tgt_input = tgt[:, :-1]
        target = tgt[:, 1:]
        
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt_input == 0)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        output, _ = model(
            src=src,
            tgt=tgt_input,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        
        # 计算损失（LabelSmoothingLoss 已忽略 pad_idx）
        output_flat = output.reshape(-1, output.size(-1))
        target_flat = target.reshape(-1)
        loss = criterion(output_flat, target_flat)
        
        # 统计非 PAD token 数量
        ntokens = (target_flat != pad_idx).sum().item()  # 注意：需传入 pad_idx 或从 criterion 获取
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD)  # 使用配置
        optimizer.step()
        
        total_loss += loss.item() * ntokens
        total_tokens += ntokens
        
        if batch_idx % 100 == 0:
            print(f"批次 {batch_idx}, 损失: {loss.item():.4f}")
    
    return total_loss / total_tokens if total_tokens > 0 else 0

def generate_square_subsequent_mask(sz):
    """生成因果掩码（防止看到未来信息）"""
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

# def validate(model, dataloader, criterion, device):
#     """验证"""
#     model.eval()
#     total_loss = 0
#     total_tokens = 0
    
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="验证", leave=False):
#             src = batch['src'].to(device)
#             tgt = batch['tgt'].to(device)
#             src_mask = batch['src_mask'].to(device)
#             tgt_mask = batch['tgt_mask'].to(device)
            
#             tgt_input = tgt[:, :-1]
#             tgt_output = tgt[:, 1:]
            
#             output, _ = model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])
            
#             loss = criterion(
#                 output.contiguous().view(-1, output.size(-1)),
#                 tgt_output.contiguous().view(-1)
#             )
            
#             batch_tokens = (tgt_output != 0).sum().item()
#             total_loss += loss.item() * batch_tokens
#             total_tokens += batch_tokens
    
#     return total_loss / total_tokens if total_tokens > 0 else 0

def validate(model, dataloader, criterion, device):
    """验证 - 与 train_epoch 使用相同的 mask 构造方式"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证", leave=False):
            src = batch['src'].to(device)  # [B, S]
            tgt = batch['tgt'].to(device)  # [B, T]
            
            tgt_input = tgt[:, :-1]        # [B, T-1]
            target = tgt[:, 1:]            # [B, T-1]
            
            # 构造 masks
            src_padding_mask = (src == 0)                     # [B, S]
            tgt_padding_mask = (tgt_input == 0)               # [B, T-1]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)  # [T-1, T-1]
            
            # 前向传播
            output, _ = model(
                src=src,
                tgt=tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else 0

def translate_example(model, sentence, en_vocab, zh_vocab, device, temperature=0.8):
    """翻译示例"""
    model.eval()
    
    # 编码输入
    src_indices = en_vocab.encode(sentence, add_special_tokens=True)
    src = torch.tensor(src_indices).unsqueeze(0).to(device)
    src_mask = torch.ones(1, 1, 1, len(src_indices)).bool().to(device)
    
    # 编码器输出
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        
        # 初始化目标序列
        tgt_indices = [zh_vocab.word2idx[Config.SOS_TOKEN]]
        
        for i in range(Config.MAX_LENGTH):
            tgt = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            
            # 创建因果掩码
            tgt_len = len(tgt_indices)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0).bool().to(device)
            tgt_mask = tgt_mask & (tgt != zh_vocab.word2idx[Config.PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
            
            # 解码
            decoder_output, _ = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            output = model.output_layer(decoder_output)
            
            # 应用温度采样
            output = output / temperature
            probs = F.softmax(output[0, -1], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            tgt_indices.append(next_token)
            
            # 遇到EOS则停止
            if next_token == zh_vocab.word2idx[Config.EOS_TOKEN]:
                break
        
        # 解码为文本
        translation = zh_vocab.decode(tgt_indices[1:-1])
        
        return translation

def main():
    """主训练函数"""
    print("🚀 Transformer翻译模型训练")
    Config.display()
    
    # 创建目录
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 设备
    device = Config.DEVICE
    print(f"使用设备: {device}")
    
    # 1. 加载词汇表
    print("\n🔤 加载词汇表...")
    en_vocab = Vocabulary.load(f"{Config.VOCAB_DIR}/en_vocab.pkl")
    zh_vocab = Vocabulary.load(f"{Config.VOCAB_DIR}/zh_vocab.pkl")
    
    print(f"英文词汇表: {len(en_vocab)}")
    print(f"中文词汇表: {len(zh_vocab)}")
    
    # 2. 创建数据加载器
    train_loader, val_loader = create_dataloaders(en_vocab, zh_vocab, Config)

    for batch in train_loader:
        print(f"src shape: {batch['src'].shape}")  # 应该是 [batch_size, seq_len]
        print(f"tgt shape: {batch['tgt'].shape}")
        break    

    # 3. 构建模型
    print("\n🏗️  构建模型...")
    model = build_model(len(en_vocab), len(zh_vocab), device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 4. 损失函数和优化器
    # pad_idx = en_vocab.word2idx[Config.PAD_TOKEN]
    pad_idx = zh_vocab.word2idx[Config.PAD_TOKEN]
    criterion = LabelSmoothingLoss(
        len(zh_vocab),
        padding_idx=pad_idx,
        smoothing=Config.LABEL_SMOOTHING
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.INIT_LR,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    if Config.LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_MAX, T_mult=2
        )
    else:
        def lr_lambda(step):
            if step < Config.WARMUP_STEPS:
                return float(step) / float(max(1, Config.WARMUP_STEPS))
            else:
                progress = (step - Config.WARMUP_STEPS) / (Config.EPOCHS * len(train_loader) - Config.WARMUP_STEPS)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 5. TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{Config.LOG_DIR}/{timestamp}")
    
    # 6. 早停
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA
    )
    
    # 7. 训练循环
    print("\n🔥 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{Config.EPOCHS}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, pad_idx)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印结果
        print(f"\n📊 Epoch {epoch} 结果:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  时间: {epoch_time:.1f}秒")
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  🎯 新的最佳验证损失!")
        
        if epoch % 5 == 0 or is_best:
            # 构建可序列化的配置
            config_save = {
                k: v for k, v in Config.__dict__.items()
                if not k.startswith('__') and isinstance(v, (int, float, str, bool))
            }
            # 手动处理 DEVICE
            config_save['DEVICE'] = str(Config.DEVICE)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                # 'en_vocab': en_vocab,
                # 'zh_vocab': zh_vocab,
                'config': config_save
            }
            
            if is_best:
                torch.save(checkpoint, f"{Config.CHECKPOINT_DIR}/best_model.pth")
                print(f"  💾 保存最佳模型")
            
            if epoch % 20 == 0:
                torch.save(checkpoint, f"{Config.CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth")
                print(f"  💾 保存检查点")
        
        # 每5轮显示翻译示例
        # if epoch % 5 == 0:
        #     print(f"\n🔍 翻译示例:")
        #     test_sentences = [
        #         "A dog is running in the park.",
        #         "Two people are building a snow house.",
        #         "A woman is cooking in the kitchen."
        #     ]
            
        #     for sentence in test_sentences:
        #         try:
        #             translation = translate_example(model, sentence, en_vocab, zh_vocab, device)
        #             print(f"  '{sentence}'")
        #             print(f"  -> '{translation}'")
        #         except Exception as e:
        #             print(f"  翻译失败: {e}")
        
        # 早停检查
        if early_stopping(val_loss):
            print(f"\n⚠️  早停触发，在 epoch {epoch}")
            break
    
    writer.close()
    
    print(f"\n🎉 训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()