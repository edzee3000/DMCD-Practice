import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
from tqdm import tqdm
import gc  # 垃圾回收

# ====================== 1. 配置参数 ======================
# 数据路径
DATA_DIR = "./Data/UserData"
TRAIN_PATH = os.path.join(DATA_DIR, "subscribe_train.txt")
TEST_PATH = os.path.join(DATA_DIR, "subscribe_test.txt")
CACHE_DIR = os.path.join("./Data", "Data_Cache")

# 标签传播参数
MAX_ITER = 5
ALPHA = 0.8
NEIGHBOR_TOPK = 20
THRESHOLD = 0.5
BATCH_SIZE = 100000  # 分批次处理训练集（关键：降低内存占用）

# ====================== 2. 加载预处理数据 ======================
def load_cached_dict():
    dict_paths = {
        "a_user": os.path.join(CACHE_DIR, "a_user_dict.pkl"),
        "b_user": os.path.join(CACHE_DIR, "b_user_dict.pkl"),
        "interact": os.path.join(CACHE_DIR, "interact_dict.pkl"),
        "subscribe": os.path.join(CACHE_DIR, "subscribe_dict.pkl"),
        "keyword": os.path.join(CACHE_DIR, "keyword_dict.pkl"),
        "vocab": os.path.join(CACHE_DIR, "vocab_dict.pkl")
    }
    
    cached_dicts = {}
    for name, path in dict_paths.items():
        with open(path, "rb") as f:
            cached_dicts[name] = pickle.load(f)
        print(f"{name}_dict 加载完成")
    return cached_dicts

# ====================== 3. 优化版：构建社交图（分批次+稀疏存储） ======================
def build_social_graph_optimized(train_path, subscribe_dict, interact_dict):
    """
    优化点：
    1. 分批次加载训练集，避免一次性加载7300万条数据
    2. 稀疏存储：仅保留有标签的(A,B)对，且用整数ID替代字符串（减少内存）
    3. 提前计算最大交互值，避免二次遍历
    4. 实时垃圾回收，释放临时内存
    """
    # 步骤1：预计算最大交互值（仅遍历一次interact_dict）
    max_interact = 1.0
    for (u, v), interact in interact_dict.items():
        weight = interact["@num"] + interact["forward_num"] + interact["comment_num"]
        if weight > max_interact:
            max_interact = weight
    print(f"最大交互值：{max_interact}")

    # 步骤2：构建边权重（稀疏存储，仅保留非零权重）
    edge_weights = defaultdict(float)
    # 交互关系权重
    for (u, v), interact in tqdm(interact_dict.items(), desc="构建交互边权重"):
        weight = (interact["@num"] + interact["forward_num"] + interact["comment_num"]) / max_interact
        edge_weights[(u, v)] = weight
    # 关注关系权重（设为1.0）
    for u, follow_set in tqdm(subscribe_dict.items(), desc="构建关注边权重"):
        for v in follow_set:
            edge_weights[(u, v)] = 1.0
    print(f"边权重构建完成，总边数：{len(edge_weights)}")

    # 步骤3：分批次加载训练集标签（核心优化）
    train_labels = {}  # 仅存储非空标签，且用整数ID
    user_id_map = {}  # 字符串ID -> 整数ID（进一步减少内存）
    next_id = 0

    def get_int_id(user_str):
        """将字符串用户ID转为整数ID"""
        nonlocal next_id
        if user_str not in user_id_map:
            user_id_map[user_str] = next_id
            next_id += 1
        return user_id_map[user_str]

    # 分批次读取训练集
    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        batch = []
        for line_idx, line in enumerate(tqdm(f, desc="分批次加载训练集")):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            A_id_str, B_id_str, label_str = parts[0], parts[1], parts[2]
            try:
                label = int(label_str)
                # 转为整数ID
                A_id = get_int_id(A_id_str)
                B_id = get_int_id(B_id_str)
                batch.append((A_id, B_id, label))
            except ValueError:
                continue

            # 批次满了就写入，释放内存
            if len(batch) >= BATCH_SIZE:
                for (A, B, l) in batch:
                    train_labels[(A, B)] = l
                batch.clear()
                gc.collect()  # 强制垃圾回收

        # 处理最后一批
        if batch:
            for (A, B, l) in batch:
                train_labels[(A, B)] = l
            batch.clear()
            gc.collect()

    print(f"训练集标签加载完成，有效标签数：{len(train_labels)}")
    print(f"用户ID映射数：{len(user_id_map)}")

    # 步骤4：构建邻接表（稀疏+采样）
    adjacency = defaultdict(list)
    for (u_str, v_str), weight in tqdm(edge_weights.items(), desc="构建邻接表"):
        # 转为整数ID
        u = get_int_id(u_str)
        v = get_int_id(v_str)
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))  # 无向图

    # 邻居采样（仅保留TOPK）
    for u in tqdm(adjacency.keys(), desc="邻居采样"):
        adjacency[u] = sorted(adjacency[u], key=lambda x: x[1], reverse=True)[:NEIGHBOR_TOPK]

    # 释放临时变量内存
    del edge_weights
    gc.collect()

    return adjacency, train_labels, user_id_map

# ====================== 4. 优化版：标签传播（增量更新+内存复用） ======================
class LabelPropagationOptimized:
    def __init__(self, adjacency, train_labels, user_id_map, alpha=ALPHA, max_iter=MAX_ITER):
        self.adjacency = adjacency
        self.train_labels = train_labels  # 仅存储有标签的(A,B)
        self.user_id_map = user_id_map
        self.alpha = alpha
        self.max_iter = max_iter
        # 稀疏存储标签分布：仅存储有更新的(A,B)对
        self.label_dist = defaultdict(lambda: {1: 0.0, -1: 0.0})

        # 初始化标注样本
        for (A, B), label in tqdm(self.train_labels.items(), desc="初始化标签分布"):
            self.label_dist[(A, B)][label] = 1.0
            self.label_dist[(A, B)][-label] = 0.0
        gc.collect()

    def propagate(self):
        """增量更新标签分布，避免全量遍历"""
        for iter_num in range(self.max_iter):
            print(f"\n迭代 {iter_num+1}/{self.max_iter}")
            new_label_dist = defaultdict(lambda: {1: 0.0, -1: 0.0})
            # 仅遍历有标签/有邻居的(A,B)对（稀疏更新）
            target_pairs = list(self.label_dist.keys())
            avg_confidence = 0.0

            for idx, (A, B) in enumerate(tqdm(target_pairs, desc="更新标签分布")):
                # 保留自身标签
                new_label_dist[(A, B)][1] = self.alpha * self.label_dist[(A, B)][1]
                new_label_dist[(A, B)][-1] = self.alpha * self.label_dist[(A, B)][-1]

                # 聚合邻居标签
                neighbor_sum = {1: 0.0, -1: 0.0}
                total_weight = 1e-6

                # 聚合A的邻居
                for (A_nei, weight_A) in self.adjacency.get(A, []):
                    if (A_nei, B) in self.label_dist:
                        neighbor_sum[1] += weight_A * self.label_dist[(A_nei, B)][1]
                        neighbor_sum[-1] += weight_A * self.label_dist[(A_nei, B)][-1]
                        total_weight += weight_A

                # 聚合B的邻居
                for (B_nei, weight_B) in self.adjacency.get(B, []):
                    if (A, B_nei) in self.label_dist:
                        neighbor_sum[1] += weight_B * self.label_dist[(A, B_nei)][1]
                        neighbor_sum[-1] += weight_B * self.label_dist[(A, B_nei)][-1]
                        total_weight += weight_B

                # 融合邻居标签
                new_label_dist[(A, B)][1] += (1 - self.alpha) * neighbor_sum[1] / total_weight
                new_label_dist[(A, B)][-1] += (1 - self.alpha) * neighbor_sum[-1] / total_weight

                # 计算置信度
                avg_confidence += max(new_label_dist[(A, B)][1], new_label_dist[(A, B)][-1])

                # 每10000条清理一次内存
                if idx % 10000 == 0:
                    gc.collect()

            # 更新标签分布
            self.label_dist = new_label_dist
            avg_confidence /= len(target_pairs)
            print(f"迭代 {iter_num+1} 平均置信度：{avg_confidence:.4f}")
            gc.collect()  # 迭代后清理内存

    def predict(self, test_pairs, user_id_map):
        """测试集预测（转为整数ID）"""
        predictions = {}
        # 先将测试集字符串ID转为整数ID
        test_int_pairs = []
        test_str2int = {}
        for (A_str, B_str) in tqdm(test_pairs, desc="转换测试集ID"):
            A = user_id_map.get(A_str, -1)
            B = user_id_map.get(B_str, -1)
            test_int_pairs.append((A, B))
            test_str2int[(A, B)] = (A_str, B_str)

        # 预测
        for (A, B) in tqdm(test_int_pairs, desc="预测测试集"):
            if A == -1 or B == -1:
                # 冷启动：未知用户默认-1
                predictions[test_str2int[(A, B)]] = -1
            elif (A, B) in self.label_dist:
                p_pos = self.label_dist[(A, B)][1]
                predictions[test_str2int[(A, B)]] = 1 if p_pos > THRESHOLD else -1
            else:
                predictions[test_str2int[(A, B)]] = -1

        return predictions

# ====================== 5. 加载测试集（分批次） ======================
def load_test_pairs_optimized(test_path, batch_size=BATCH_SIZE):
    test_pairs = []
    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        batch = []
        for line in tqdm(f, desc="加载测试集"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            A_id, B_id = parts[0], parts[1]
            batch.append((A_id, B_id))
            if len(batch) >= batch_size:
                test_pairs.extend(batch)
                batch.clear()
        if batch:
            test_pairs.extend(batch)
    return test_pairs

# ====================== 6. 评估函数 ======================
def evaluate_predictions(predictions):
    pos_ratio = sum(1 for v in predictions.values() if v == 1) / len(predictions)
    print(f"预测为1（关注）的比例：{pos_ratio:.4f}")
    print(f"总预测数：{len(predictions)}")
    return {"pos_ratio": pos_ratio}

# ====================== 7. 主流程 ======================
def main():
    # 强制垃圾回收
    gc.enable()
    start_time = time.time()

    # 1. 加载缓存
    print("加载缓存字典...")
    cached_dicts = load_cached_dict()
    subscribe_dict = cached_dicts["subscribe"]
    interact_dict = cached_dicts["interact"]

    # 2. 构建优化版社交图
    print("\n构建优化版社交图...")
    adjacency, train_labels, user_id_map = build_social_graph_optimized(
        TRAIN_PATH, subscribe_dict, interact_dict
    )
    # 释放无用字典内存
    del cached_dicts
    gc.collect()

    # 3. 初始化标签传播模型
    print("\n初始化标签传播模型...")
    lp_model = LabelPropagationOptimized(
        adjacency, train_labels, user_id_map, alpha=ALPHA, max_iter=MAX_ITER
    )
    # 释放邻接表内存（模型内已保存）
    del adjacency
    gc.collect()

    # 4. 执行传播
    print("\n开始标签传播...")
    lp_model.propagate()

    # 5. 加载测试集
    print("\n加载测试集...")
    test_pairs = load_test_pairs_optimized(TEST_PATH)

    # 6. 预测
    print("\n预测测试集...")
    predictions = lp_model.predict(test_pairs, user_id_map)

    # 7. 评估
    print("\n预测结果统计：")
    evaluate_predictions(predictions)

    # 8. 保存结果（分批次保存，避免内存溢出）
    print("\n保存预测结果...")
    save_path = os.path.join(DATA_DIR, "lp_predictions_optimized.pkl")
    # 分批次保存
    with open(save_path, "wb") as f:
        pickle.dump(dict(list(predictions.items())[:BATCH_SIZE]), f)
        for i in range(BATCH_SIZE, len(predictions), BATCH_SIZE):
            pickle.dump(dict(list(predictions.items())[i:i+BATCH_SIZE]), f)
    print(f"预测结果已保存到 {save_path}")

    # 总耗时
    total_time = (time.time() - start_time) / 60
    print(f"\n总耗时：{total_time:.2f} 分钟")

if __name__ == "__main__":
    main()