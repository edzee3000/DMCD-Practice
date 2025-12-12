import sys
import pickle
import os
import pandas as pd
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练（可选）
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 训练参数
STREAM_BATCH_SIZE = 131072  # 流式读取TXT的批次大小（可根据内存调整）
TRAIN_BATCH_SIZE = 1024      # 模型训练批次大小
EPOCHS = 5                 # 训练轮数
LEARNING_RATE = 0.001      # 学习率
# 定义字典保存路径（建议统一放在data_cache目录）
view_num = 2
UserPath = "./Data/UserData"
CACHE_DIR = "./Data/Data_Cache"
TRAIN_DATA_PATH = f"{UserPath}/subscribe_train.txt"
A_USER_DICT_PATH = f"{CACHE_DIR}/a_user_dict.pkl"
B_USER_DICT_PATH = f"{CACHE_DIR}/b_user_dict.pkl"
INTERACT_DICT_PATH = f"{CACHE_DIR}/interact_dict.pkl"
SUBSCRIBE_DICT_PATH = f"{CACHE_DIR}/subscribe_dict.pkl"
KEYWORD_DICT_PATH = f"{CACHE_DIR}/keyword_dict.pkl"
VOCAB_DICT_PATH = f"{CACHE_DIR}/vocab_dict.pkl"  # 词汇表也一起保存
# ---------------------- 字典序列化/反序列化工具函数 ----------------------
def save_dict_to_file(target_dict, file_path):
    """
    将字典保存到本地文件
    :param target_dict: 要保存的字典对象
    :param file_path: 保存路径（如 "a_user_dict.pkl"）
    """
    # 创建父目录（若不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        # 高协议版本提升读写速度
        pickle.dump(target_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"字典已保存到：{file_path}")

def load_dict_from_file(file_path):
    """
    从本地文件读取字典
    :param file_path: 字典文件路径
    :return: 反序列化后的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"字典文件不存在：{file_path}")
    with open(file_path, "rb") as f:
        target_dict = pickle.load(f)
    print(f"字典已从 {file_path} 加载完成")
    return target_dict

# ---------------------- 核心工具函数：读取无表头TXT文件（制表符分隔） ----------------------
def read_txt_no_header(file_path, sep="\t", chunk_size=10000):
    """
    读取无表头TXT文件，逐块返回数据列表（避免DataFrame的表头依赖）
    :param file_path: TXT文件路径
    :param sep: 分隔符，默认制表符\t
    :param chunk_size: 每次读取的行数
    :return: 生成器，每次返回chunk_size行的二维列表
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        chunk = []
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 按制表符分割，处理不规则分隔（如多个\t）
            row = [x for x in line.split(sep) if x]
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:  # 处理最后一块不足chunk_size的数据
            yield chunk


def stream_txt_file(file_path, sep="\t", batch_size=8192):
    """
    流式读取TXT文件，逐批次返回数据（不落地内存）
    :param file_path: TXT文件路径
    :param sep: 分隔符
    :param batch_size: 每批次行数
    :return: 生成器，每次返回batch_size行的二维列表
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        batch = []
        for line_num, line in enumerate(f):
            line = line.strip()
            # 过滤空行/无效行
            if not line or len(line) < 5:
                continue
            # 分割行数据，处理不规则分隔
            row = [x.strip() for x in line.split(sep) if x.strip()]
            # 过滤字段不足的异常行（训练集至少4列：A_id,B_id,label,timestamp）
            if len(row) < 4:
                continue
            # 过滤标签异常行（仅保留1/-1）
            if row[2] not in ["1", "-1"]:
                continue
            batch.append(row)
            # 达到批次大小则返回
            if len(batch) >= batch_size:
                yield batch
                batch = []
        # 处理最后一批不足batch_size的数据
        if batch:
            yield batch


# ---------------------- 多值特征处理工具函数 ----------------------
def parse_multi_value(val, sep=";", dtype=str):
    """
    解析分号分隔的多值特征，返回列表（空值返回["0"]）
    """
    if val == 0 or pd.isna(val) or val == "0":
        return ["0"]
    return [dtype(x.strip()) for x in str(val).split(sep) if x.strip()]

def parse_kw_weight(val, sep=";"):
    """
    解析关键词:权重对（如183:0.6673;2:0.3535），返回字典{关键词ID: 权重}
    """
    if val == 0 or pd.isna(val) or val == "0":
        return {"0": 0.0}
    kw_dict = {}
    for item in str(val).split(sep):
        if ":" not in item:
            continue
        kw_id, weight = item.split(":", 1)
        try:
            kw_dict[str(kw_id.strip())] = float(weight.strip())
        except:
            kw_dict[str(kw_id.strip())] = 0.0
    return kw_dict

# 数一下训练集样本个数
def count_train_samples(train_txt_path, save_path=f"{CACHE_DIR}/train_sample_count.pkl"):
    """
    统计训练集有效样本数（仅执行一次，结果缓存到本地）
    :param train_txt_path: 训练集TXT路径
    :param save_path: 样本数缓存路径
    :return: 有效样本总数
    """
    # 优先读取缓存
    try:
        with open(save_path, "rb") as f:
            count = pickle.load(f)
        print(f"训练集样本数（从缓存读取）：{count}")
        return count
    except FileNotFoundError:
        print("正在统计训练集有效样本数...")
        count = 0
        # 流式遍历训练集，统计有效行
        for batch in stream_txt_file(train_txt_path):
            count += len(batch)
        # 保存统计结果到缓存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(count, f)
        print(f"训练集有效样本数：{count}")
        return count