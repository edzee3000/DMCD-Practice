from utils import *

class RecommendIterableDataset(IterableDataset):
    def __init__(self, train_txt_path, a_user_dict, b_user_dict, interact_dict, subscribe_dict, keyword_dict,
                 gender_vocab, tag_vocab, cate_vocab, kw_vocab, batch_size=10000):
        self.train_txt_path = train_txt_path
        self.a_user_dict = a_user_dict
        self.b_user_dict = b_user_dict
        self.interact_dict = interact_dict
        self.subscribe_dict = subscribe_dict
        self.keyword_dict = keyword_dict
        self.gender_vocab = gender_vocab
        self.tag_vocab = tag_vocab
        self.cate_vocab = cate_vocab
        self.kw_vocab = kw_vocab
        self.batch_size = batch_size  # 流式读取的批次大小

    def _extract_feat_safe(self, A_id, B_id):
        """
        安全提取特征：所有异常值/缺失值均补0，保证返回合法特征
        :return: wide_feat, deep_feat, label（均为numpy数组，无NaN/Inf）
        """
        # 初始化特征为0（兜底）
        wide_feat = np.zeros(8, dtype=np.float32)
        deep_feat = np.zeros(16, dtype=np.float32)
        
        try:
            # ---------------------- Wide特征提取（8维） ----------------------
            # 1. A性别编码
            A_gender = self.a_user_dict.get(A_id, {}).get("gender", 0)
            wide_feat[0] = self.gender_vocab.get(str(A_gender), 0)
            # 2. B一级分类编码
            B_cate_1 = self.b_user_dict.get(B_id, {}).get("cate_1", "0")
            wide_feat[1] = self.cate_vocab.get(B_cate_1, 0)
            # 3. A与B关键词交集占比
            A_kw_dict = self.keyword_dict.get(A_id, {"0": 0.0})
            B_kw_list = self.b_user_dict.get(B_id, {}).get("keyword_ids", ["0"])
            kw_inter = len(set(A_kw_dict.keys()) & set(B_kw_list))
            wide_feat[2] = min(kw_inter / len(B_kw_list), 1.0) if len(B_kw_list) > 0 else 0.0
            # 4. A是否关注过B
            A_followed = self.subscribe_dict.get(A_id, set())
            wide_feat[3] = 1 if B_id in A_followed else 0
            # 5. 交互行为总数归一化
            interact_info = self.interact_dict.get((A_id, B_id), {"@num":0, "forward_num":0, "comment_num":0})
            total_inter = interact_info["@num"] + interact_info["forward_num"] + interact_info["comment_num"]
            wide_feat[4] = min(total_inter / 100, 1.0)
            # 6. A标签数归一化
            A_tag_num = len(self.a_user_dict.get(A_id, {}).get("tag_ids", ["0"]))
            wide_feat[5] = min(A_tag_num / 20, 1.0)
            # 7. B关键词数归一化
            B_kw_num = len(self.b_user_dict.get(B_id, {}).get("keyword_ids", ["0"]))
            wide_feat[6] = min(B_kw_num / 50, 1.0)
            # 8. A发帖数归一化
            A_post_num = self.a_user_dict.get(A_id, {}).get("post_num", 0.0)
            wide_feat[7] = min(A_post_num / 1000, 1.0)

            # ---------------------- Deep特征提取（16维） ----------------------
            # 数值特征（前8维，对数变换+防Inf）
            deep_feat[0] = np.log1p(max(A_post_num, 0))  # 防负数
            deep_feat[1] = np.log1p(max(interact_info["@num"], 0))
            deep_feat[2] = np.log1p(max(interact_info["forward_num"], 0))
            deep_feat[3] = np.log1p(max(interact_info["comment_num"], 0))
            deep_feat[4] = np.log1p(max(sum(A_kw_dict.values()), 0))
            deep_feat[5] = np.log1p(max(A_tag_num, 0))
            deep_feat[6] = np.log1p(max(B_kw_num, 0))
            deep_feat[7] = np.log1p(max(total_inter, 0))
            # 离散特征索引（后8维）
            deep_feat[8] = self.gender_vocab.get(str(A_gender), 0)
            deep_feat[9] = self.tag_vocab.get(self.a_user_dict.get(A_id, {}).get("tag_ids", ["0"])[0], 0)
            deep_feat[10] = self.cate_vocab.get(B_cate_1, 0)
            deep_feat[11] = self.cate_vocab.get(self.b_user_dict.get(B_id, {}).get("cate_2", "0"), 0)
            deep_feat[12] = self.cate_vocab.get(self.b_user_dict.get(B_id, {}).get("cate_3", "0"), 0)
            deep_feat[13] = self.cate_vocab.get(self.b_user_dict.get(B_id, {}).get("cate_4", "0"), 0)
            deep_feat[14] = self.kw_vocab.get(self.b_user_dict.get(B_id, {}).get("keyword_ids", ["0"])[0], 0)
            deep_feat[15] = self.kw_vocab.get(list(A_kw_dict.keys())[0], 0)

            # 数值特征归一化（防NaN/Inf）
            numeric_part = deep_feat[:8]
            numeric_part = (numeric_part - np.nanmin(numeric_part)) / (np.nanmax(numeric_part) - np.nanmin(numeric_part) + 1e-8)
            deep_feat[:8] = numeric_part
            # 替换NaN/Inf为0
            wide_feat = np.nan_to_num(wide_feat, nan=0.0, posinf=0.0, neginf=0.0)
            deep_feat = np.nan_to_num(deep_feat, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            # 任意异常均返回全0特征（兜底）
            pass
        return wide_feat, deep_feat

    def __iter__(self):
        """
        迭代器核心逻辑：流式读取训练集，逐行提取特征，返回单条样本
        """
        # 流式读取训练集
        for batch in stream_txt_file(self.train_txt_path, batch_size=self.batch_size):
            for row in batch:
                A_id = str(row[0])
                B_id = str(row[1])
                label = 1 if row[2] == "1" else 0
                # 安全提取特征
                wide_feat, deep_feat = self._extract_feat_safe(A_id, B_id)
                # 转换为Tensor并返回
                yield (
                    torch.tensor(wide_feat, dtype=torch.float32),
                    torch.tensor(deep_feat, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.float32)
                )
