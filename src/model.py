"""Transformer模型"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# class TransformerModel(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
#                  num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
#                  dropout=0.1, max_seq_length=5000, device=None):
#         super(TransformerModel, self).__init__()
#         self.d_model = d_model
#         self.device = device if device else torch.device('cpu')
#         # 词嵌入
#         self.src_embedding = nn.Embedding(src_vocab_size, d_model)
#         self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
#         # 位置编码
#         self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
#         # Transformer
#         self.transformer = nn.Transformer(
#             d_model=d_model,
#             nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True  # 重要：使用 batch_first=True
#         )
        
#         # 输出层
#         self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
#     def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, 
#                 memory_key_padding_mask=None, tgt_mask=None):
#         # 嵌入 + 位置编码
#         src_emb = self.positional_encoding(self.src_embedding(src))
#         tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        
#         # 通过Transformer
#         output = self.transformer(
#             src_emb,
#             tgt_emb,
#             tgt_mask=tgt_mask,
#             src_key_padding_mask=src_key_padding_mask,
#             tgt_key_padding_mask=tgt_key_padding_mask,
#             memory_key_padding_mask=memory_key_padding_mask
#         )
        
#         # 输出层
#         output = self.fc_out(output)
#         return output, None  # 保持与原始代码兼容

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, max_seq_length=5000):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer - 使用 batch_first=True 简化处理
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 重要！
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None,
                memory_mask=None):
        """
        参数:
        - src: 源语言序列 [batch_size, src_len]
        - tgt: 目标语言序列 [batch_size, tgt_len]
        - src_mask: 源语言的自注意力掩码 [src_len, src_len]
        - tgt_mask: 目标语言的因果掩码 [tgt_len, tgt_len]
        - src_padding_mask: 源语言的填充掩码 [batch_size, src_len]
        - tgt_padding_mask: 目标语言的填充掩码 [batch_size, tgt_len]
        - memory_mask: 解码器的交叉注意力掩码 [tgt_len, src_len]
        """
        
        # 嵌入 + 位置编码
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 通过Transformer
        output = self.transformer(
            src=src_emb,  # 源序列
            tgt=tgt_emb,  # 目标序列（输入到解码器）
            src_mask=src_mask,  # 源语言的自注意力掩码
            tgt_mask=tgt_mask,  # 目标语言的因果掩码
            memory_mask=memory_mask,  # 解码器的交叉注意力掩码
            src_key_padding_mask=src_padding_mask,  # 源语言的填充掩码
            tgt_key_padding_mask=tgt_padding_mask,  # 目标语言的填充掩码
        )
        
        # 输出层
        output = self.fc_out(output)
        return output, None  # 保持与原始代码兼容
    
class Transformer(nn.Module):
    """Transformer模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, device=None):
        super().__init__()
        
        self.device = device if device else torch.device('cpu')
        self.d_model = d_model
        
        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask):
        """编码器"""
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        memory = self.transformer.encoder(src_embedded, src_key_padding_mask=~src_mask.squeeze(1).squeeze(1))
        return memory
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        """解码器"""
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        output = self.transformer.decoder(
            tgt_embedded, memory,
            tgt_key_padding_mask=~tgt_mask.squeeze(1).squeeze(1),
            memory_key_padding_mask=~src_mask.squeeze(1).squeeze(1)
        )
        return output, None
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """前向传播"""
        # 编码
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        
        # 解码
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Transformer
        output = self.transformer(
            src_embedded, tgt_embedded,
            src_key_padding_mask=~src_mask.squeeze(1).squeeze(1),
            tgt_key_padding_mask=~tgt_mask.squeeze(1).squeeze(1),
            memory_key_padding_mask=~src_mask.squeeze(1).squeeze(1)
        )
        
        # 输出层
        output = self.output_layer(output)
        
        return output, None

def build_model(src_vocab_size, tgt_vocab_size, device, **kwargs):
    """构建模型"""
    from config import Config
    
    # 使用配置或覆盖参数
    d_model = kwargs.get('d_model', Config.D_MODEL)
    nhead = kwargs.get('nhead', Config.N_HEAD)
    num_encoder_layers = kwargs.get('num_encoder_layers', Config.NUM_ENCODER_LAYERS)
    num_decoder_layers = kwargs.get('num_decoder_layers', Config.NUM_DECODER_LAYERS)
    dim_feedforward = kwargs.get('dim_feedforward', Config.DIM_FEEDFORWARD)
    dropout = kwargs.get('dropout', Config.DROPOUT)
    
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    
    return model.to(device)