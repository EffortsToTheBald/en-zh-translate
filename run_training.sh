#!/bin/bash

# 训练脚本

echo "🚀 开始训练流程"
echo "=" * 60

# 1. 构建词汇表
echo "📝 步骤1: 构建词汇表"
python scripts/build_vocab.py

# 2. 开始训练
echo -e "\n📝 步骤2: 开始训练"
python src/train.py

# 3. 测试模型
echo -e "\n📝 步骤3: 测试模型"
python src/inference.py

echo -e "\n✅ 训练完成!"