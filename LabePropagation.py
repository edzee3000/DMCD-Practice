import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
from tqdm import tqdm

# ====================== 1. 配置参数 ======================
# 数据路径
DATA_DIR = "./Data/UserData"
TRAIN_PATH = os.path.join(DATA_DIR, "subscribe_train.txt")
TEST_PATH = os.path.join(DATA_DIR, "subscribe_test.txt")
CACHE_DIR = os.path.join("./Data", "Data_Cache")

# 标签传播参数
MAX_ITER = 5  # 传播迭代次数
ALPHA = 0.8   # 传播权重（保留自身标签的概率）
NEIGHBOR_TOPK = 20  # 稀疏图邻居采样数（解决冷启动）
THRESHOLD = 0.5     # 最终分类阈值（>0.5为1，否则为-1）

# ====================== 2. 加载预处理数据（复用原有字典） ======================
def load_cached_dict():
    """加载已构建的特征字典"""
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

# ====================== 3. 构建社交网络图 ======================
def build_social_graph(train_path, subscribe_dict, interact_dict):
    """
    构建社交网络图：
    - 节点：所有用户ID（A类+B类）
    - 边：关注关系（subscribe）+ 交互关系（interact），带权重
    - 边权重：交互频次（@+转发+评论）/ 最大交互值（归一化）
    """
    # 1. 加载训练集标签
    train_labels = {}  # (A_id, B_id) -> label (1/-1)
    max_interact = 1.0  # 归一化用的最大交互值
    edge_weights = defaultdict(float)  # (u, v) -> weight
    
    # 2. 统计交互权重
    for (u, v), interact in interact_dict.items():
        weight = interact["@num"] + interact["forward_num"] + interact["comment_num"]
        edge_weights[(u, v)] = weight
        if weight > max_interact:
            max_interact = weight
    
    # 3. 归一化边权重 + 融合关注关系
    for (u, v), weight in edge_weights.items():
        edge_weights[(u, v)] = weight / max_interact  # 交互权重归一化到[0,1]
    # 关注关系权重设为1.0（强关系）
    for u, follow_set in subscribe_dict.items():
        for v in follow_set:
            edge_weights[(u, v)] = 1.0  # 关注关系权重高于交互
    
    # 4. 加载训练集标签
    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="加载训练集标签"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            A_id, B_id, label = parts[0], parts[1], int(parts[2])
            train_labels[(A_id, B_id)] = label
    
    # 5. 构建邻接表（稀疏图优化：仅保留TOPK邻居）
    adjacency = defaultdict(list)
    for (u, v), weight in edge_weights.items():
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))  # 无向图（交互/关注双向传播）
    
    # 6. 邻居采样（解决大规模稀疏图）
    for u in adjacency:
        # 按权重降序排序，取TOPK
        adjacency[u] = sorted(adjacency[u], key=lambda x: x[1], reverse=True)[:NEIGHBOR_TOPK]
    
    return adjacency, edge_weights, train_labels

# ====================== 4. 标签传播算法实现 ======================
class LabelPropagation:
    def __init__(self, adjacency, train_labels, alpha=ALPHA, max_iter=MAX_ITER):
        self.adjacency = adjacency  # 邻接表 {u: [(v, weight)]}
        self.train_labels = train_labels  # 标注样本 (A,B)->label
        self.alpha = alpha  # 保留自身标签的概率
        self.max_iter = max_iter  # 传播迭代次数
        # 初始化标签分布：(A,B) -> 标签概率分布 {1: p, -1: 1-p}
        self.label_dist = defaultdict(lambda: {1: 0.0, -1: 0.0})
        
        # 初始化标注样本的标签分布
        for (A, B), label in train_labels.items():
            self.label_dist[(A, B)][label] = 1.0
            self.label_dist[(A, B)][-label] = 0.0
    
    def propagate(self):
        """标签传播核心逻辑：迭代更新未标注样本的标签分布"""
        for iter_num in range(self.max_iter):
            print(f"\n标签传播迭代 {iter_num+1}/{self.max_iter}")
            new_label_dist = defaultdict(lambda: {1: 0.0, -1: 0.0})
            
            # 遍历所有待预测的(A,B)对
            for (A, B) in tqdm(self.label_dist.keys(), desc="更新标签分布"):
                # 1. 保留自身原有标签（若有）
                new_label_dist[(A, B)][1] = self.alpha * self.label_dist[(A, B)][1]
                new_label_dist[(A, B)][-1] = self.alpha * self.label_dist[(A, B)][-1]
                
                # 2. 聚合邻居的标签分布（加权平均）
                neighbor_sum = {1: 0.0, -1: 0.0}
                total_weight = 1e-6  # 避免除零
                
                # 聚合A的邻居对B的标签
                for (A_nei, weight_A) in self.adjacency.get(A, []):
                    if (A_nei, B) in self.label_dist:
                        neighbor_sum[1] += weight_A * self.label_dist[(A_nei, B)][1]
                        neighbor_sum[-1] += weight_A * self.label_dist[(A_nei, B)][-1]
                        total_weight += weight_A
                
                # 聚合B的邻居被A的标签
                for (B_nei, weight_B) in self.adjacency.get(B, []):
                    if (A, B_nei) in self.label_dist:
                        neighbor_sum[1] += weight_B * self.label_dist[(A, B_nei)][1]
                        neighbor_sum[-1] += weight_B * self.label_dist[(A, B_nei)][-1]
                        total_weight += weight_B
                
                # 3. 融合邻居标签（1-alpha）
                new_label_dist[(A, B)][1] += (1 - self.alpha) * neighbor_sum[1] / total_weight
                new_label_dist[(A, B)][-1] += (1 - self.alpha) * neighbor_sum[-1] / total_weight
            
            # 更新标签分布
            self.label_dist = new_label_dist
            # 打印迭代信息
            avg_confidence = np.mean([max(d[1], d[-1]) for d in self.label_dist.values()])
            print(f"迭代 {iter_num+1} 平均置信度：{avg_confidence:.4f}")
    
    def predict(self, test_pairs):
        """预测测试集标签：根据标签分布取最大值"""
        predictions = {}
        for (A, B) in tqdm(test_pairs, desc="预测测试集"):
            if (A, B) in self.label_dist:
                p_pos = self.label_dist[(A, B)][1]
                predictions[(A, B)] = 1 if p_pos > THRESHOLD else -1
            else:
                # 冷启动处理：无邻居信息时，默认预测-1
                predictions[(A, B)] = -1
        return predictions

# ====================== 5. 加载测试集 & 评估结果 ======================
def load_test_pairs(test_path):
    """加载测试集待预测的(A,B)对"""
    test_pairs = []
    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="加载测试集"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            A_id, B_id = parts[0], parts[1]
            test_pairs.append((A_id, B_id))
    return test_pairs

def evaluate_predictions(predictions, true_labels=None):
    """评估预测结果（若有真实标签）"""
    if true_labels is None:
        print("无真实标签，仅输出预测结果统计")
        pos_ratio = sum(1 for v in predictions.values() if v == 1) / len(predictions)
        print(f"预测为1（关注）的比例：{pos_ratio:.4f}")
        return
    
    # 计算准确率/召回率/F1
    tp = tn = fp = fn = 0
    for (A, B), pred in predictions.items():
        true = true_labels.get((A, B), -1)
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == -1:
            fp += 1
        elif pred == -1 and true == 1:
            fn += 1
        elif pred == -1 and true == -1:
            tn += 1
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
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
    # 1. 加载缓存数据
    cached_dicts = load_cached_dict()
    subscribe_dict = cached_dicts["subscribe"]
    interact_dict = cached_dicts["interact"]
    
    # 2. 构建社交图
    print("\n开始构建社交网络图...")
    adjacency, edge_weights, train_labels = build_social_graph(
        TRAIN_PATH, subscribe_dict, interact_dict
    )
    print(f"社交图构建完成：节点数={len(adjacency)}, 边数={len(edge_weights)}")
    
    # 3. 初始化标签传播模型
    print("\n初始化标签传播模型...")
    lp_model = LabelPropagation(adjacency, train_labels, alpha=ALPHA, max_iter=MAX_ITER)
    
    # 4. 执行标签传播
    print("\n开始标签传播...")
    lp_model.propagate()
    
    # 5. 加载测试集并预测
    print("\n加载测试集...")
    test_pairs = load_test_pairs(TEST_PATH)
    predictions = lp_model.predict(test_pairs)
    
    # 6. 输出结果
    print("\n预测结果统计：")
    evaluate_predictions(predictions)
    
    # 7. 保存预测结果
    with open(os.path.join(DATA_DIR, "lp_predictions.pkl"), "wb") as f:
        pickle.dump(predictions, f)
    print("\n预测结果已保存到 lp_predictions.pkl")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n总耗时：{(time.time() - start_time)/60:.2f} 分钟")