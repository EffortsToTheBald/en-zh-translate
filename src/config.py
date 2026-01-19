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
    VOCAB_DIR = "vocab_new"
    
    # 模型参数
    D_MODEL = 256
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.2
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 50
    INIT_LR = 0.0001
    WARMUP_STEPS = 4000
    WEIGHT_DECAY = 0.0001
    CLIP_GRAD = 1.0
    
    # 学习率调度
    LR_SCHEDULER = "cosine"
    T_MAX = 10
    
    # 数据参数
    MAX_LENGTH = 50
    MAX_VOCAB = 4000
    
    # 标签平滑
    LABEL_SMOOTHING = 0.1
    
    # 特殊标记
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    
    # 目录
    CHECKPOINT_DIR = "checkpoints_new"
    LOG_DIR = "logs_new"
    RESULTS_DIR = "results_new"
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    
    # 早停
    PATIENCE = 10
    MIN_DELTA = 0.001
    
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