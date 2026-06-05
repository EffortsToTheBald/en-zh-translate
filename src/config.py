"""配置文件"""
import torch

class Config:
    """训练配置"""
    
    # 数据路径
    TRAIN_EN_FILE = "data/train.en"
    TRAIN_ZH_FILE = "data/train.zh"
    VAL_EN_FILE = "data/val.en"
    VAL_ZH_FILE = "data/val.zh"
    
    # 词汇表路径
    VOCAB_DIR = "vocab_test"
    
    # 模型参数
    D_MODEL = 512
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1536
    DROPOUT = 0.15
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 1 #80
    INIT_LR = 0.00001
    MAX_LR = 0.0003
    WARMUP_STEPS = 3000 # 总步数 ≈ 100 * 480 = 48,000 → 5% 是 2400，但小数据集可更激进（1000 足够）
    WEIGHT_DECAY = 0.0005
    CLIP_GRAD = 1
    
    # 学习率调度
    LR_SCHEDULER = "transformer"
    T_MAX = 100
    
    # 数据参数
    MAX_LENGTH = 80
    MAX_VOCAB_ZH = 16000
    MAX_VOCAB = 28000
    
    # 标签平滑
    LABEL_SMOOTHING = 0.15
    
    # 特殊标记
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 3
    UNK_IDX = 2
    SOS_IDX = 1
    EOS_IDX = 0

    # 目录
    CHECKPOINT_DIR = "checkpoints_new"
    LOG_DIR = "logs_new"
    RESULTS_DIR = "results_new"
    MODEL_DIR = "model-ga/model"
    VAL_DIR = "model-ga/val"
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    
    # 早停
    PATIENCE = 20
    MIN_DELTA = 0.0001
    
    @classmethod
    def display(cls):
        """显示配置"""
        print("📋 训练配置:")
        print("=" * 60)
        print(f"  训练数据: {cls.TRAIN_EN_FILE}")
        print(f"  验证数据: {cls.VAL_EN_FILE}")
        print(f"  模型大小: d_model={cls.D_MODEL}, layers={cls.NUM_ENCODER_LAYERS}")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  训练轮数: {cls.EPOCHS}")
        print(f"  学习率: {cls.INIT_LR}")
        print(f"  设备: {cls.DEVICE}")
        print("=" * 60)

    @classmethod
    def init_token_ids(cls, en_tokenizer, zh_tokenizer):
        """从 tokenizer 动态获取特殊 token ID"""
        cls.PAD_IDX = en_tokenizer.piece_to_id("<pad>")
        cls.UNK_IDX = en_tokenizer.piece_to_id("<unk>")
        cls.SOS_IDX = en_tokenizer.piece_to_id("<s>")
        cls.EOS_IDX = en_tokenizer.piece_to_id("</s>")
        
        # 验证中英文 tokenizer 一致性
        assert cls.PAD_IDX == zh_tokenizer.piece_to_id("<pad>")
        assert cls.SOS_IDX == zh_tokenizer.piece_to_id("<s>")
        assert cls.EOS_IDX == zh_tokenizer.piece_to_id("</s>")
        print(f"✅ Token IDs: PAD={cls.PAD_IDX}, SOS={cls.SOS_IDX}, EOS={cls.EOS_IDX}, UNK={cls.UNK_IDX}")