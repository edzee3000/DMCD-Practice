import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
from tqdm import tqdm
import psutil

# ====================== 1. 配置参数 ======================
# 数据路径（确保目录存在）
DATA_DIR = "./Data/UserData"
CACHE_DIR = os.path.join("./Data", "Data_Cache")
TRAIN_PATH = os.path.join(DATA_DIR, "subscribe_train.txt")
TEST_PATH = os.path.join(DATA_DIR, "subscribe_test.txt")
OUTPUT_PATH = os.path.join(DATA_DIR, "lp_predictions.pkl")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 标签传播参数
MAX_ITER = 5  # 传播迭代次数
ALPHA = 0.8   # 传播权重（保留自身标签的概率）
NEIGHBOR_TOPK = 20  # 稀疏图邻居采样数
THRESHOLD = 0.5     # 最终分类阈值
BATCH_SIZE = 10000  # 测试集分批处理大小

# ====================== 2. 加载预处理数据 ======================
def load_cached_dict():
    """加载    加载已构建的特征字典
    若缓存不存在则创建空字典（避免首次运行报错）
    """
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
        if os.path.exists(path):
            with open(path, "rb") as f:
                cached_dicts[name] = pickle.load(f)
            print(f"{name}_dict 加载完成 (大小: {os.path.getsize(path)/1024:.2f}KB)")
        else:
            # 处理空缓存（首次运行场景）
            if name == "interact":
                cached_dicts[name] = defaultdict(dict)
            elif name == "subscribe":
                cached_dicts[name] = defaultdict(set)
            else:
                cached_dicts[name] = dict()
            print(f"{name}_dict 不存在，使用空字典初始化")
    return cached_dicts

# ====================== 3. 构建社交网络图 ======================
def build_social_graph(train_path, subscribe_dict, interact_dict):
    """构建社交网络图并优化内存使用"""
    # 1. 初始化数据结构
    train_labels = {}  # (A_id, B_id) -> label (1/-1)
    max_interact = 1.0
    edge_weights = defaultdict(float)
    
    # 2. 统计交互权重（仅保留有效数据）
    for (u, v), interact in interact_dict.items():
        if not isinstance(interact, dict):
            continue  # 过滤无效数据
        try:
            weight = interact.get("@num", 0) + interact.get("forward_num", 0) + interact.get("comment_num", 0)
        except (TypeError, KeyError):
            continue  # 跳过格式错误数据
        if weight > 0:
            edge_weights[(u, v)] = weight
            if weight > max_interact:
                max_interact = weight
    
    # 3. 归一化边权重 + 融合关注关系
    for (u, v) in list(edge_weights.keys()):  # 使用list避免迭代中修改
        edge_weights[(u, v)] /= max_interact
    
    # 关注关系权重设为1.0（强关系）
    for u, follow_set in subscribe_dict.items():
        for v in follow_set:
            edge_weights[(u, v)] = 1.0  # 覆盖交互权重
    
    # 4. 加载训练集标签（容错处理）
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="加载训练集标签"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                try:
                    A_id, B_id, label = parts[0], parts[1], int(parts[2])
                    if label in (1, -1):
                        train_labels[(A_id, B_id)] = label
                except (ValueError, IndexError):
                    continue  # 跳过格式错误行
    else:
        print(f"警告：训练集文件 {train_path} 不存在，使用空训练集")

    # 5. 构建邻接表（优化存储：使用元组列表而非defaultdict(list)）
    adjacency = dict()  # 改为普通字典减少内存开销
    for (u, v), weight in edge_weights.items():
        # 处理u的邻居
        if u not in adjacency:
            adjacency[u] = []
        adjacency[u].append((v, weight))
        # 处理v的邻居（无向图）
        if v not in adjacency:
            adjacency[v] = []
        adjacency[v].append((u, weight))
    
    # 6. 邻居采样（按权重排序取TOPK）
    for u in adjacency:
        # 排序并截断，同时去重
        seen = set()
        unique_neighbors = []
        for v, w in adjacency[u]:
            if v not in seen:
                seen.add(v)
                unique_neighbors.append((v, w))
        # 按权重降序排序取TOPK
        unique_neighbors.sort(key=lambda x: x[1], reverse=True)
        adjacency[u] = unique_neighbors[:NEIGHBOR_TOPK]
    
    return adjacency, edge_weights, train_labels

# ====================== 4. 标签传播算法实现（内存优化版） ======================
class LabelPropagation:
    def __init__(self, adjacency, train_labels, alpha=ALPHA, max_iter=MAX_ITER):
        self.adjacency = adjacency
        self.alpha = alpha
        self.max_iter = max_iter
        self.pair_list = list(train_labels.keys())  # 所有待处理的(A,B)对
        self.pair_index = {pair: i for i, pair in enumerate(self.pair_list)}  # 索引映射
        self.n_pairs = len(self.pair_list)
        
        # 用numpy数组存储1的概率（-1的概率=1-p，节省空间）
        self.label_probs = np.zeros(self.n_pairs, dtype=np.float32)
        for i, (A, B) in enumerate(self.pair_list):
            self.label_probs[i] = 1.0 if train_labels[(A, B)] == 1 else 0.0
        
        # 预分配新概率数组（复用内存）
        self.new_label_probs = np.zeros_like(self.label_probs)

    def propagate(self):
        """标签传播核心逻辑（优化内存和计算效率）"""
        for iter_num in range(self.max_iter):
            print(f"\n标签传播迭代 {iter_num+1}/{self.max_iter}")
            self.new_label_probs[:] = self.alpha * self.label_probs  # 保留自身标签
            
            # 遍历所有待预测的(A,B)对
            for i in tqdm(range(self.n_pairs), desc="更新标签分布"):
                A, B = self.pair_list[i]
                neighbor_sum = 0.0
                total_weight = 1e-6  # 避免除零
                
                # 聚合A的邻居对B的标签
                for (A_nei, weight_A) in self.adjacency.get(A, []):
                    neighbor_pair = (A_nei, B)
                    if neighbor_pair in self.pair_index:
                        j = self.pair_index[neighbor_pair]
                        neighbor_sum += weight_A * self.label_probs[j]
                        total_weight += weight_A
                
                # 聚合B的邻居被A的标签
                for (B_nei, weight_B) in self.adjacency.get(B, []):
                    neighbor_pair = (A, B_nei)
                    if neighbor_pair in self.pair_index:
                        j = self.pair_index[neighbor_pair]
                        neighbor_sum += weight_B * self.label_probs[j]
                        total_weight += weight_B
                
                # 融合邻居标签
                self.new_label_probs[i] += (1 - self.alpha) * (neighbor_sum / total_weight)
            
            # 交换数组引用（避免内存分配）
            self.label_probs, self.new_label_probs = self.new_label_probs, self.label_probs
            
            # 打印迭代信息
            avg_confidence = np.mean(np.maximum(self.label_probs, 1 - self.label_probs))
            print(f"迭代 {iter_num+1} 平均置信度：{avg_confidence:.4f}")
            print(f"当前内存使用：{psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    def predict_batch(self, batch):
        """批量预测测试集"""
        predictions = {}
        for (A, B) in batch:
            if (A, B) in self.pair_index:
                idx = self.pair_index[(A, B)]
                p_pos = self.label_probs[idx]
                predictions[(A, B)] = 1 if p_pos > THRESHOLD else -1
            else:
                predictions[(A, B)] = -1  # 冷启动默认值
        return predictions

# ====================== 5. 测试集处理与评估 ======================
def load_test_pairs_batch(test_path, batch_size=BATCH_SIZE):
    """分批加载测试集（减少内存占用）"""
    if not os.path.exists(test_path):
        print(f"警告：测试集文件 {test_path} 不存在，返回空列表")
        yield []
        return
        
    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        batch = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                batch.append((parts[0], parts[1]))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

def evaluate_predictions(predictions, true_labels=None):
    """评估预测结果"""
    if not predictions:
        print("无预测结果可评估")
        return
    
    if true_labels is None:
        pos_ratio = sum(1 for v in predictions.values() if v == 1) / len(predictions)
        print(f"预测为1（关注）的比例：{pos_ratio:.4f}")
        return
    
    # 计算评估指标
    tp = tn = fp = fn = 0
    for (A, B), pred in predictions.items():
        true = true_labels.get((A, B), -1)
        if pred == 1:
            if true == 1:
                tp += 1
            else:
                fp += 1
        else:
            if true == 1:
                fn += 1
            else:
                tn += 1
    
    # 避免除零错误
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"准确率：{accuracy:.4f}")
    print(f"精确率：{precision:.4f}")
    print(f"召回率：{recall:.4f}")
    print(f"F1分数：{f1:.4f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# ====================== 6. 主流程执行 ======================
def main():
    start_time = time.time()
    print("===== 标签传播算法开始执行 =====")
    
    # 1. 加载缓存数据
    cached_dicts = load_cached_dict()
    subscribe_dict = cached_dicts["subscribe"]
    interact_dict = cached_dicts["interact"]
    
    # 2. 构建社交图
    print("\n开始构建社交网络图...")
    adjacency, edge_weights, train_labels = build_social_graph(
        TRAIN_PATH, subscribe_dict, interact_dict
    )
    print(f"社交图构建完成：节点数={len(adjacency)}, 边数={len(edge_weights)}, 训练样本数={len(train_labels)}")
    
    if not train_labels:
        print("警告：未加载到训练样本，无法进行标签传播")
        return
    
    # 3. 初始化标签传播模型
    print("\n初始化标签传播模型...")
    lp_model = LabelPropagation(adjacency, train_labels, alpha=ALPHA, max_iter=MAX_ITER)
    
    # 4. 执行标签传播
    print("\n开始标签传播...")
    lp_model.propagate()
    
    # 5. 加载测试集并分批预测
    print("\n开始预测测试集...")
    predictions = {}
    for batch in load_test_pairs_batch(TEST_PATH):
        if not batch:
            continue
        batch_preds = lp_model.predict_batch(batch)
        predictions.update(batch_preds)
    print(f"测试集预测完成：共{len(predictions)}个样本")
    
    # 6. 评估与保存结果
    print("\n预测结果统计：")
    evaluate_predictions(predictions)
    
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(predictions, f)
    print(f"\n预测结果已保存到 {OUTPUT_PATH}")
    
    # 7. 输出总耗时
    total_time = (time.time() - start_time) / 60
    print(f"\n总耗时：{total_time:.2f} 分钟")
    print("===== 标签传播算法执行完毕 =====")

if __name__ == "__main__":
    main()