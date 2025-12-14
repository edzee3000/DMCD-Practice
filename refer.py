import sys
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from Models import *
import warnings
import gc  # 导入垃圾回收模块
warnings.filterwarnings('ignore')


# ====================== 1. 核心配置（必须与训练一致） ======================
# 路径配置
DATA_DIR = "./Data/UserData"
CACHE_DIR = "./Data/Data_Cache"
TEST_PATH = os.path.join(DATA_DIR, "subscribe_test.txt")  # 你的测试集路径
MODEL_PATH = "./WideDeep_Epoch_5.pth"  # 训练好的模型权重
SUBMISSION_PATH = "./Data/submission.csv"    # 最终输出文件（匹配目标格式）

# 模型参数（和训练时完全一致！）
WIDE_DIM = 8               # Wide部分维度
DEEP_NUMERIC_DIM = 8       # Deep数值特征维度
HIDDEN_DIMS = [128, 64]    # Deep部分隐藏层

# 推理参数
STREAM_BATCH_SIZE = 524288  # 流式读取测试集批次（减少内存）
# STREAM_BATCH_SIZE = 1048576  # 流式读取测试集批次（减少内存）
# STREAM_BATCH_SIZE = 2097152  # 流式读取测试集批次（减少内存）
INFER_BATCH_SIZE = 8192     # 模型推理批次（根据显存调整）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 2. 加载缓存资源（特征字典/词汇表） ======================
def load_cache():
    """加载训练时生成的缓存文件，保证特征一致"""
    cache_files = {
        "a_user": "a_user_dict.pkl",
        "b_user": "b_user_dict.pkl",
        "interact": "interact_dict.pkl",
        "subscribe": "subscribe_dict.pkl",
        "keyword": "keyword_dict.pkl",
        "vocab": "vocab_dict.pkl"
    }
    cache = {}
    for name, filename in cache_files.items():
        path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缓存文件缺失：{path}")
        with open(path, "rb") as f:
            cache[name] = pickle.load(f)
        print(f"✅ {name} 缓存加载完成")
    return cache

# ====================== 3. 测试集数据集（流式读取+特征提取） ======================
class TestDataset(IterableDataset):
    def __init__(self, test_path, cache):
        self.test_path = test_path
        self.cache = cache
        # 提取特征字典
        self.a_user = cache["a_user"]
        self.b_user = cache["b_user"]
        self.interact = cache["interact"]
        self.subscribe = cache["subscribe"]
        self.keyword = cache["keyword"]
        self.vocab = cache["vocab"]

    def _parse_line(self, line):
        """解析测试集行：A_id\tB_id\t0\ttimestamp → 返回A_id, B_id"""
        line = line.strip()
        if not line:
            return None, None
        parts = line.split("\t")
        if len(parts) < 2:
            return None, None
        return parts[0], parts[1]

    def _extract_feat(self, A_id, B_id):
        """提取与训练完全一致的特征（Wide+Deep）"""
        # ---------------- Wide特征（8维） ----------------
        wide = np.zeros(WIDE_DIM, dtype=np.float32)
        # 1. A用户性别
        a_gender = self.a_user.get(A_id, {}).get("gender", 0)
        wide[0] = self.vocab["gender_vocab"].get(str(a_gender), 0)
        # 2. B用户一级分类
        b_cate1 = self.b_user.get(B_id, {}).get("cate_1", "0")
        wide[1] = self.vocab["cate_vocab"].get(b_cate1, 0)
        # 3. 关键词交集占比
        a_kw = set(self.keyword.get(A_id, {}).keys())
        b_kw = set(self.b_user.get(B_id, {}).get("keyword_ids", []))
        wide[2] = len(a_kw & b_kw) / max(len(b_kw), 1)
        # 4. A是否关注B
        wide[3] = 1.0 if B_id in self.subscribe.get(A_id, set()) else 0.0
        # 5. 交互次数归一化
        inter = self.interact.get((A_id, B_id), {"@num":0, "forward_num":0, "comment_num":0})
        total_inter = inter["@num"] + inter["forward_num"] + inter["comment_num"]
        wide[4] = min(total_inter / 100, 1.0)
        # 6. A用户标签数量归一化
        a_tag_num = len(self.a_user.get(A_id, {}).get("tag_ids", []))
        wide[5] = min(a_tag_num / 20, 1.0)
        # 7. B用户关键词数量归一化
        b_kw_num = len(self.b_user.get(B_id, {}).get("keyword_ids", []))
        wide[6] = min(b_kw_num / 50, 1.0)
        # 8. A用户发帖数归一化
        a_post_num = self.a_user.get(A_id, {}).get("post_num", 0)
        wide[7] = min(a_post_num / 1000, 1.0)

        # ---------------- Deep特征（16维） ----------------
        deep = np.zeros(16, dtype=np.float32)
        # 数值特征（前8维）
        deep[0] = np.log1p(a_post_num)
        deep[1] = np.log1p(inter["@num"])
        deep[2] = np.log1p(inter["forward_num"])
        deep[3] = np.log1p(inter["comment_num"])
        deep[4] = np.log1p(sum(self.keyword.get(A_id, {}).values()))
        deep[5] = np.log1p(a_tag_num)
        deep[6] = np.log1p(b_kw_num)
        deep[7] = np.log1p(total_inter)
        # 离散特征（后8维，转为词汇表ID）
        deep[8] = self.vocab["gender_vocab"].get(str(a_gender), 0)
        deep[9] = self.vocab["tag_vocab"].get(self.a_user.get(A_id, {}).get("tag_ids", ["0"])[0], 0)
        deep[10] = self.vocab["cate_vocab"].get(b_cate1, 0)
        deep[11] = self.vocab["cate_vocab"].get(self.b_user.get(B_id, {}).get("cate_2", "0"), 0)
        deep[12] = self.vocab["cate_vocab"].get(self.b_user.get(B_id, {}).get("cate_3", "0"), 0)
        deep[13] = self.vocab["cate_vocab"].get(self.b_user.get(B_id, {}).get("cate_4", "0"), 0)
        deep[14] = self.vocab["kw_vocab"].get(self.b_user.get(B_id, {}).get("keyword_ids", ["0"])[0], 0)
        deep[15] = self.vocab["kw_vocab"].get(list(a_kw)[0] if a_kw else "0", 0)

        # 特征归一化+异常值处理
        wide = np.nan_to_num(wide, 0.0)
        deep[:8] = (deep[:8] - deep[:8].min()) / (deep[:8].max() - deep[:8].min() + 1e-8)
        deep = np.nan_to_num(deep, 0.0)
        return wide, deep

    def __iter__(self):
        """流式迭代：返回 (wide_feat, deep_feat, A_id, B_id)"""
        with open(self.test_path, "r", encoding="utf-8", errors="ignore") as f:
            batch = []
            for line in f:
                A_id, B_id = self._parse_line(line)
                if not A_id or not B_id:
                    continue
                wide, deep = self._extract_feat(A_id, B_id)
                batch.append((wide, deep, A_id, B_id))
                # 批次满则返回
                if len(batch) >= STREAM_BATCH_SIZE:
                    for item in batch:
                        yield item
                    batch = []
            # 处理最后一批
            for item in batch:
                yield item



# ====================== 5. 核心推理逻辑（模型输出→目标格式） ======================
def main():
    # 1. 加载缓存和模型
    cache = load_cache()
    vocab = cache["vocab"]  # 从缓存中获取词汇表
    EMBED_CONFIG = [
        (len(vocab["gender_vocab"]), 4),
        (len(vocab["tag_vocab"]), 8),
        (len(vocab["cate_vocab"]), 8),
        (len(vocab["cate_vocab"]), 8),
        (len(vocab["cate_vocab"]), 8),
        (len(vocab["cate_vocab"]), 8),
        (len(vocab["kw_vocab"]), 8),
        (len(vocab["kw_vocab"]), 8)
    ]
    # 初始化模型
    model = WideDeep(
        wide_dim=WIDE_DIM,
        deep_numeric_dim=DEEP_NUMERIC_DIM,
        embed_config=EMBED_CONFIG,
        hidden_dims=HIDDEN_DIMS
    ).to(DEVICE)
    # 加载权重（兼容CPU/GPU）
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()  # 推理模式（关闭Dropout）
    print(f"✅ 模型加载完成，设备：{DEVICE}")

    # 2. 初始化测试集和DataLoader
    test_dataset = TestDataset(TEST_PATH, cache)
    test_loader = DataLoader(
        test_dataset,
        batch_size=INFER_BATCH_SIZE,
        num_workers=0,
        pin_memory=True
    )

    # 3. 推理：收集每个用户A的候选B及得分
    user_candidates = {}  # key: A_id, value: [(B_id, score), ...]
    total_processed = 0  # 新增：用于统计已处理的样本总数
    with torch.no_grad():  # 禁用梯度，节省内存
        pbar = tqdm(test_loader, desc="推理中", unit="batch")
        for batch in pbar:
            wide_feat, deep_feat, A_ids, B_ids = batch
            batch_size = len(A_ids)  # 当前批次样本数
            # 数据转tensor并移到设备
            wide_feat = torch.tensor(np.stack(wide_feat)).to(DEVICE)
            deep_feat = torch.tensor(np.stack(deep_feat)).to(DEVICE)
            # 模型预测（logits→概率）
            logits = model(wide_feat, deep_feat)
            scores = torch.sigmoid(logits).cpu().numpy()  # 转为0-1的关注概率

            # 按用户A分组存储
            for A_id, B_id, score in zip(A_ids, B_ids, scores):
                if A_id not in user_candidates:
                    user_candidates[A_id] = []
                user_candidates[A_id].append((B_id, score))
            
            # # 关键优化：删除当前批次变量，释放内存
            # del wide_feat, deep_feat, logits, scores, A_ids, B_ids
            # # torch.cuda.empty_cache()  # 释放GPU缓存（如果用GPU）
            # gc.collect()  # 强制垃圾回收
            
            # 累计样本数，并更新进度条状态
            total_processed += batch_size
            # 将已处理样本数添加到进度条的后缀信息中（显示在[]内）
            pbar.set_postfix(samples=f"{total_processed}")
    # 保存user_candidates到缓存  防止后面代码出错
    CANDIDATES_CACHE_PATH = os.path.join(CACHE_DIR, "user_candidates.pkl")
    os.makedirs(os.path.dirname(CANDIDATES_CACHE_PATH), exist_ok=True)
    with open(CANDIDATES_CACHE_PATH, "wb") as f:
        pickle.dump(user_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ user_candidates已保存到：{CANDIDATES_CACHE_PATH}")
    # 从缓存中重新读取，替换原有变量
    with open(CANDIDATES_CACHE_PATH, "rb") as f:
        user_candidates = pickle.load(f)
    print(f"✅ 已从缓存加载user_candidates，共包含 {len(user_candidates)} 个用户")

    # 4. 生成目标格式的提交文件
    with open(SUBMISSION_PATH, "w", encoding="utf-8") as f:
        # 写入表头（匹配示例：id,clicks）
        f.write("id,clicks\n")
        # 遍历每个用户，生成TOP3推荐
        for A_id in tqdm(user_candidates.keys(), desc="生成结果"):
            # 按得分降序排序 → 去重 → 取前3
            candidates = sorted(user_candidates[A_id], key=lambda x: x[1], reverse=True)
            unique_B = []
            seen = set()
            for B_id, _ in candidates:
                if B_id not in seen:
                    seen.add(B_id)
                    unique_B.append(B_id)
                if len(unique_B) >= 3:
                    break
            # 拼接为空格分隔的字符串（匹配示例格式）
            clicks = " ".join(unique_B) if unique_B else ""
            # 写入行（id,clicks）
            f.write(f"{A_id},{clicks}\n")

    print(f"\n✅ 推理完成！结果已保存到：{SUBMISSION_PATH}")
    print(f"📊 统计：共处理 {len(user_candidates)} 个用户")

if __name__ == "__main__":
    main()