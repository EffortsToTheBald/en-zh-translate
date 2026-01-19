"""
使用 SentencePiece Tokenizer 的 Transformer 训练脚本
适用于中英机器翻译任务
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import math
from datetime import datetime

# 添加项目根目录到路径（便于模块导入）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import LabelSmoothingLoss, EarlyStopping

def get_transformer_lr(step: int, d_model: int, warmup_steps: int, max_lr: float = 0.0005):
    """原论文学习率公式"""
    if step == 0:
        return 1e-8
    lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return min(lr, max_lr)

def generate_square_subsequent_mask(sz):
    """生成 decoder 所需的下三角 mask，防止看到未来词"""
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, pad_idx,global_step):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        # 先更新学习率，再执行训练步骤（修复时机问题）
        lr = get_transformer_lr(
            step=global_step,
            d_model=Config.D_MODEL,
            warmup_steps=Config.WARMUP_STEPS,
            max_lr=Config.MAX_LR
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if global_step % 480 == 0:
            theoretical_lr = get_transformer_lr(global_step, Config.D_MODEL, Config.WARMUP_STEPS)
            actual_lr = optimizer.param_groups[0]['lr']
            print(f"  [DEBUG] Step {global_step}: theoretical={theoretical_lr:.6f}, actual={actual_lr:.6f}")

        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_padding_mask = (src == pad_idx).to(device)
        tgt_padding_mask = (tgt_input == pad_idx).to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        output, _ = model(
            src=src,
            tgt=tgt_input,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask  
        )

        output_flat = output.view(-1, output.size(-1))
        tgt_flat = tgt_output.reshape(-1)

        loss = criterion(output_flat, tgt_flat)
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Loss is NaN or Inf! Skipping batch.")
            optimizer.zero_grad()
            continue
        ntokens = (tgt_flat != pad_idx).sum().item()

        optimizer.zero_grad()

        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD, norm_type=2.0)
        # 梯度爆炸保护：如果裁剪后的梯度范数仍然过大，跳过更新
        if total_norm > 10:  # 阈值可根据情况调整
            print(f"⚠️ 梯度爆炸! Norm={total_norm:.2f}，跳过本轮更新")
            optimizer.zero_grad()
            global_step += 1
            continue
        # elif total_norm > Config.CLIP_GRAD:
        #     print(f"⚠️ 梯度裁剪! Norm={total_norm:.2f} (阈值={Config.CLIP_GRAD})")
        optimizer.step()

        global_step += 1

        total_loss += loss.item() * ntokens
        total_tokens += ntokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss , global_step

def validate(model, val_loader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_padding_mask = (src == pad_idx).to(device)
            tgt_padding_mask = (tgt_input == pad_idx).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            output, _ = model(
                src=src,
                tgt=tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )

            output_flat = output.view(-1, output.size(-1))
            tgt_flat = tgt_output.reshape(-1)

            loss = criterion(output_flat, tgt_flat)
            ntokens = (tgt_flat != pad_idx).sum().item()

            total_loss += loss.item() * ntokens
            total_tokens += ntokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss

def main():
    """主训练函数"""
    print("🚀 启动 Transformer 中英翻译训练（SentencePiece 版）")
    Config.display()
    
    # 创建目录
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 设备
    device = Config.DEVICE
    print(f"🔧 使用设备: {device}")
    
    # Step 1: 加载数据 + tokenizer
    print("\n📂 加载数据集和 Tokenizer...")
    train_loader, val_loader, en_tokenizer, zh_tokenizer = create_dataloaders(Config)
    print("🔍 Tokenizer 调试:")
    print(f"  EN PAD: '{en_tokenizer.id_to_piece(Config.PAD_IDX)}' (ID={Config.PAD_IDX})")
    print(f"  ZH PAD: '{zh_tokenizer.id_to_piece(Config.PAD_IDX)}' (ID={Config.PAD_IDX})")
    print(f"  EN SOS: '{en_tokenizer.id_to_piece(Config.SOS_IDX)}'")
    print(f"  ZH EOS: '{zh_tokenizer.id_to_piece(Config.EOS_IDX)}'")    
    assert zh_tokenizer.id_to_piece(Config.PAD_IDX) == "<pad>", "中文 PAD ID 错误！"
    Config.init_token_ids(en_tokenizer, zh_tokenizer)
    # Step 2: 构建模型
    print("\n🏗️  构建模型...")
    src_vocab_size = en_tokenizer.vocab_size
    tgt_vocab_size = zh_tokenizer.vocab_size
    print(f"英文词汇表: {src_vocab_size}")
    print(f"中文词汇表: {tgt_vocab_size}")

    model = build_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: 总 {total_params:,} | 可训练 {trainable_params:,}")
    
    # Step 3: 损失函数 & 优化器
    pad_idx = Config.PAD_IDX  
    criterion = LabelSmoothingLoss(
        tgt_vocab_size,
        padding_idx=pad_idx,
        smoothing=Config.LABEL_SMOOTHING
    )

    # def lr_lambda(current_step: int):
    #     """Transformer 原版 LR 调度（每步调用）"""
    #     if current_step == 0:
    #         return 1e-8

    #     lr = (Config.D_MODEL ** -0.5) * min(
    #     current_step ** -0.5,
    #     current_step * (Config.WARMUP_STEPS ** -1.5)
    #     )
    #     return min(lr, 0.0004)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.INIT_LR,  
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=Config.WEIGHT_DECAY
    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Step 5. TensorBoard````````````````````
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{Config.LOG_DIR}/{timestamp}")
    
    # Step 6. 早停
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA,
        verbose=True
    )
    
    # Step 7. 训练循环
    print("\n🔥 开始训练...")
    best_val_loss = float('inf')
    global_step = 0
    print("✅ Starting training with global_step = 0")   
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n📅 开始第 {epoch}/{Config.EPOCHS} 轮训练")
        
        start_time = time.time()
        
        # ✅ 调用 train_epoch 时传入 scheduler，并接收 lr
        train_loss, global_step  = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, pad_idx,global_step 
        )        
                
        # 验证
        val_loss = validate(model, val_loader, criterion, device, pad_idx)        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # 打印结果
        print(f"\n📊 Epoch {epoch} 结果:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  学习率: {current_lr:.8f}")
        print(f"  时间: {epoch_time:.1f}秒")
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  🎯 新的最佳验证损失!")
        
        if is_best:
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
                # 'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config_save
            }
            
            if is_best:
                torch.save(checkpoint, f"{Config.CHECKPOINT_DIR}/best_model.pth")
                print(f"  💾 保存最佳模型")
        
        
        # 早停检查
        if early_stopping(val_loss):
            print(f"\n⚠️  早停触发，在 epoch {epoch}")
            break
    
    writer.close()
    
    print(f"\n🎉 训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()