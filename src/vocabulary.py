"""词汇表管理模块"""
import pickle
from collections import Counter
import re

class Vocabulary:
    """词汇表类"""
    
    def __init__(self, language):
        self.language = language
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # 特殊标记
        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"
        self.UNK_TOKEN = "<unk>"
        
        # 添加特殊标记
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.SOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)
    
    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def tokenize_en(self, text):
        """英文分词"""
        text = text.lower().strip()
        # 处理常见缩写
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        
        # 分词
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def tokenize_zh(self, text):
        """中文分词（字符级+常用词）"""
        common_words = [
            '一个', '一群', '正在', '穿着', '坐在', '站在', '看着', '戴着', '拿着',
            '蓝色', '红色', '绿色', '黄色', '白色', '黑色', '男人', '女人', '男孩',
            '女孩', '孩子', '小狗', '小猫', '汽车', '卡车', '火车', '飞机', '自行车',
            '房间', '厨房', '客厅', '卧室', '餐厅', '学校', '公园', '湖边', '山上',
            '街道', '马路', '沙滩', '海边'
        ]
        
        text = text.strip()
        tokens = []
        i = 0
        
        while i < len(text):
            matched = False
            
            # 首先尝试匹配常用词
            for word in common_words:
                if text.startswith(word, i):
                    tokens.append(word)
                    i += len(word)
                    matched = True
                    break
            
            if not matched:
                # 单个字符
                char = text[i]
                if char.strip():
                    tokens.append(char)
                i += 1
        
        return tokens
    
    def encode(self, text, add_special_tokens=True):
        """将文本编码为索引序列"""
        if self.language == "en":
            tokens = self.tokenize_en(text)
        else:
            tokens = self.tokenize_zh(text)
        
        indices = []
        if add_special_tokens:
            indices.append(self.word2idx[self.SOS_TOKEN])
        
        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]))
        
        if add_special_tokens:
            indices.append(self.word2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        tokens = []
        for idx in indices:
            if idx == self.word2idx[self.PAD_TOKEN]:
                continue
            if idx in [self.word2idx[self.SOS_TOKEN], self.word2idx[self.EOS_TOKEN]]:
                continue
            
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            tokens.append(word)
        
        if self.language == "zh":
            return "".join(tokens)
        else:
            return " ".join(tokens)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        """保存词汇表到文件"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """从文件加载词汇表"""
        with open(path, 'rb') as f:
            return pickle.load(f)