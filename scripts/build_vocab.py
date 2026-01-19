"""构建词汇表脚本"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vocabulary import Vocabulary
from src.config import Config
from collections import Counter
from tqdm import tqdm

def build_vocabularies():
    """构建词汇表"""
    print("🔤 构建词汇表")
    print("=" * 60)
    
    # 创建词汇表
    en_vocab = Vocabulary("en")
    zh_vocab = Vocabulary("zh")
    
    # 基础词汇
    base_en = [
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'in', 'on', 'at', 'by', 'with', 'to', 'from', 'for', 'of',
        'and', 'or', 'but', 'so', 'because', 'if', 'then',
        'man', 'men', 'woman', 'women', 'boy', 'girl', 'child', 'children',
        'dog', 'cat', 'animal', 'car', 'truck', 'bus', 'train', 'plane',
        'room', 'house', 'building', 'street', 'road', 'park', 'beach',
        'wearing', 'sitting', 'standing', 'walking', 'running', 'playing',
        'red', 'blue', 'green', 'yellow', 'white', 'black', 'brown',
        'small', 'big', 'large', 'little', 'old', 'young', 'new'
    ]
    
    base_zh = [
        '一个', '一群', '正在', '穿着', '戴着', '拿着', '坐在', '站在',
        '走在', '跑在', '玩在', '看着', '听着', '说着', '笑着', '哭着',
        '男人', '女人', '男孩', '女孩', '孩子', '小狗', '小猫', '动物',
        '汽车', '卡车', '公交车', '火车', '飞机', '自行车', '摩托车',
        '房间', '房子', '建筑', '街道', '马路', '公园', '沙滩', '海边',
        '红色', '蓝色', '绿色', '黄色', '白色', '黑色', '棕色', '灰色',
        '小的', '大的', '老的', '年轻的', '新的', '旧的', '长的', '短的',
        '，', '。', '！', '？'
    ]
    
    # 添加基础词
    for word in base_en:
        en_vocab.add_word(word)
    
    for word in base_zh:
        zh_vocab.add_word(word)
    
    # 统计词频
    print("📊 统计词频...")
    
    # 获取行数
    with open(Config.TRAIN_EN_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 处理训练数据
    with open(Config.TRAIN_EN_FILE, 'r', encoding='utf-8') as f_en, \
         open(Config.TRAIN_ZH_FILE, 'r', encoding='utf-8') as f_zh:
        
        for en_line, zh_line in tqdm(zip(f_en, f_zh), total=total_lines, desc="处理"):
            en_text = en_line.strip()
            zh_text = zh_line.strip()
            
            if not en_text or not zh_text:
                continue
            
            # 英文分词和统计
            en_tokens = en_vocab.tokenize_en(en_text)
            for token in en_tokens:
                en_vocab.word_freq[token] += 1
            
            # 中文分词和统计
            zh_tokens = zh_vocab.tokenize_zh(zh_text)
            for token in zh_tokens:
                zh_vocab.word_freq[token] += 1
    
    # 添加高频词
    print("🔧 添加高频英文词...")
    en_sorted = sorted(en_vocab.word_freq.items(), key=lambda x: x[1], reverse=True)
    for word, freq in en_sorted:
        if word not in en_vocab.word2idx:
            if freq >= 2:
                if len(en_vocab) < Config.MAX_VOCAB:
                    en_vocab.add_word(word)
                else:
                    break
    
    print("🔧 添加高频中文词...")
    zh_sorted = sorted(zh_vocab.word_freq.items(), key=lambda x: x[1], reverse=True)
    for word, freq in zh_sorted:
        if word not in zh_vocab.word2idx:
            if freq >= 2:
                if len(zh_vocab) < Config.MAX_VOCAB:
                    zh_vocab.add_word(word)
                else:
                    break
    
    print(f"英文词汇表: {len(en_vocab)}")
    print(f"中文词汇表: {len(zh_vocab)}")
    
    # 保存词汇表
    print("💾 保存词汇表...")
    os.makedirs(Config.VOCAB_DIR, exist_ok=True)
    en_vocab.save(f"{Config.VOCAB_DIR}/en_vocab.pkl")
    zh_vocab.save(f"{Config.VOCAB_DIR}/zh_vocab.pkl")
    
    # 测试
    print("\n🧪 测试词汇表:")
    test_cases = [
        ("A group of men are loading cotton onto a truck", "一群人把棉花装上卡车"),
        ("A man sleeping in a green room on a couch.", "一个人睡在沙发上的绿色房间"),
    ]
    
    for en, zh in test_cases:
        print(f"\n英文: '{en}'")
        en_encoded = en_vocab.encode(en)
        print(f"  编码: {en_encoded}")
        print(f"  解码: '{en_vocab.decode(en_encoded)}'")
        
        print(f"中文: '{zh}'")
        zh_encoded = zh_vocab.encode(zh)
        print(f"  编码: {zh_encoded}")
        print(f"  解码: '{zh_vocab.decode(zh_encoded)}'")

if __name__ == "__main__":
    build_vocabularies()