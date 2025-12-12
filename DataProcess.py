from utils import *
from Dataset import *






# ---------------------- 加载并清洗各数据集（构建查询字典） ----------------------
# ---------------------- 特征字典构建/加载主逻辑 ----------------------
# 1. A类用户特征：user_basic_info.txt → 列对应：user_id, birth, gender, post_num, tag_id
a_user_dict = {}
try:
    a_user_dict = load_dict_from_file(A_USER_DICT_PATH)
except FileNotFoundError:
    print("未找到A用户特征缓存，开始构建...")
    for chunk in read_txt_no_header(f"{UserPath}/user_basic_info.txt"):
        for row in chunk:
            if len(row) < 5:  # 过滤字段不足的异常行
                continue
            user_id = str(row[0])
            birth = str(row[1])
            gender = int(row[2]) if row[2].isdigit() else 0
            post_num = float(row[3]) if row[3].replace(".", "").isdigit() else 0.0
            tag_ids = parse_multi_value(row[4])
            a_user_dict[user_id] = {"birth":birth, "gender": gender, "post_num": post_num, "tag_ids": tag_ids}
    save_dict_to_file(a_user_dict, A_USER_DICT_PATH)
print(f"a_user_dict字典构造完毕  字典占用存储空间大小为: {sys.getsizeof(a_user_dict)}")
for key,value in list(a_user_dict.items())[:view_num]:
    print(f"键:{key}, 值:{value}")

# 2. B类用户特征：user_feature.txt → 列对应：user_id, category, keyword_id
b_user_dict = {}
try:
    b_user_dict = load_dict_from_file(B_USER_DICT_PATH)
except FileNotFoundError:
    print("未找到B用户特征缓存，开始构建...")
    for chunk in read_txt_no_header(f"{UserPath}/user_feature.txt"):
        for row in chunk:
            if len(row) < 3:
                continue
            user_id = str(row[0])
            # 拆分四级分类 a.b.c.d
            cate_str = row[1] if len(row)>=2 else "0"
            cate_list = cate_str.split(".") if "." in cate_str else [cate_str]
            while len(cate_list) < 4:
                cate_list.append("0")
            keyword_ids = parse_multi_value(row[2] if len(row)>=3 else "0")
            b_user_dict[user_id] = {
                "cate_1": cate_list[0], "cate_2": cate_list[1],
                "cate_3": cate_list[2], "cate_4": cate_list[3],
                "keyword_ids": keyword_ids
            }
    save_dict_to_file(b_user_dict, B_USER_DICT_PATH)
print(f"b_user_dict字典构造完毕  字典占用存储空间大小为: {sys.getsizeof(b_user_dict)}")
for key,value in list(b_user_dict.items())[:view_num]:
    print(f"键:{key}, 值:{value}")

# 3. 加载/构建用户交互特征字典
try:
    interact_dict = load_dict_from_file(INTERACT_DICT_PATH)
except FileNotFoundError:
    print("未找到用户交互缓存，开始构建...")
    interact_dict = {}
    for chunk in read_txt_no_header(f"{UserPath}/user_interaction.txt"):
        for row in chunk:
            if len(row) < 5:
                continue
            main_user = str(row[0])
            target_user = str(row[1])
            at_num = float(row[2]) if row[2].replace(".", "").isdigit() else 0.0
            forward_num = float(row[3]) if row[3].replace(".", "").isdigit() else 0.0
            comment_num = float(row[4]) if row[4].replace(".", "").isdigit() else 0.0
            interact_dict[(main_user, target_user)] = {"@num": at_num, "forward_num": forward_num, "comment_num": comment_num}
    save_dict_to_file(interact_dict, INTERACT_DICT_PATH)
print(f"interact_dict字典构造完毕  字典占用存储空间大小为: {sys.getsizeof(interact_dict)}")
for key,value in list(interact_dict.items())[:view_num]:
    print(f"键:{key}, 值:{value}")


# 4. 加载/构建用户关注历史字典
try:
    subscribe_dict = load_dict_from_file(SUBSCRIBE_DICT_PATH)
except FileNotFoundError:
    print("未找到用户关注缓存，开始构建...")
    subscribe_dict = {}
    for chunk in read_txt_no_header(f"{UserPath}/user_subscribe.txt"):
        for row in chunk:
            if len(row) < 2:
                continue
            user_id = str(row[0])
            followed_id = str(row[1])
            if user_id not in subscribe_dict:
                subscribe_dict[user_id] = set()
            subscribe_dict[user_id].add(followed_id)
    save_dict_to_file(subscribe_dict, SUBSCRIBE_DICT_PATH)
print(f"subscribe_dict字典构造完毕  字典占用存储空间大小为: {sys.getsizeof(subscribe_dict)}")
for key,value in list(subscribe_dict.items())[:view_num]:
    print(f"键:{key}, 值:{value}")

# 5. 加载/构建用户行为关键词字典
try:
    keyword_dict = load_dict_from_file(KEYWORD_DICT_PATH)
except FileNotFoundError:
    print("未找到用户关键词缓存，开始构建...")
    keyword_dict = {}
    for chunk in read_txt_no_header(f"{UserPath}/user_keyword.txt"):
        for row in chunk:
            if len(row) < 2:
                continue
            user_id = str(row[0])
            kw_weight_str = row[1]
            keyword_dict[user_id] = parse_kw_weight(kw_weight_str)
    save_dict_to_file(keyword_dict, KEYWORD_DICT_PATH)
print(f"keyword_dict字典构造完毕  字典占用存储空间大小为: {sys.getsizeof(keyword_dict)}")
for key,value in list(keyword_dict.items())[:view_num]:
    print(f"键:{key}, 值:{value}")



# 6. 加载/构建离散特征词汇表（一次性保存所有词汇表）
try:
    vocab_dict = load_dict_from_file(VOCAB_DICT_PATH)
    gender_vocab = vocab_dict["gender_vocab"]
    tag_vocab = vocab_dict["tag_vocab"]
    cate_vocab = vocab_dict["cate_vocab"]
    kw_vocab = vocab_dict["kw_vocab"]
except FileNotFoundError:
    print("未找到词汇表缓存，开始构建...")
    gender_vocab = {"0": 0, "1": 1, "2": 2}
    tag_vocab = {"0": 0}
    for feat in a_user_dict.values():
        for tag in feat["tag_ids"]:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)
    cate_vocab = {"0": 0}
    for feat in b_user_dict.values():
        for cate in [feat["cate_1"], feat["cate_2"], feat["cate_3"], feat["cate_4"]]:
            if cate not in cate_vocab:
                cate_vocab[cate] = len(cate_vocab)
    kw_vocab = {"0": 0}
    for feat in b_user_dict.values():
        for kw in feat["keyword_ids"]:
            if kw not in kw_vocab:
                kw_vocab[kw] = len(kw_vocab)
    for kw_dict in keyword_dict.values():
        for kw in kw_dict.keys():
            if kw not in kw_vocab:
                kw_vocab[kw] = len(kw_vocab)
    # 合并所有词汇表为一个字典保存
    vocab_dict = {
        "gender_vocab": gender_vocab,
        "tag_vocab": tag_vocab,
        "cate_vocab": cate_vocab,
        "kw_vocab": kw_vocab
    }
    save_dict_to_file(vocab_dict, VOCAB_DICT_PATH)
print(f"所有特征字典加载完成！")
print(f"词汇表规模：性别{len(gender_vocab)} | 标签{len(tag_vocab)} | 分类{len(cate_vocab)} | 关键词{len(kw_vocab)}")




train_sample_count = count_train_samples(f"{TRAIN_DATA_PATH}")
# print(f"训练集有效样本总数：{train_sample_count}")
# 初始化流式数据集
train_dataset = RecommendIterableDataset(
    train_txt_path=TRAIN_DATA_PATH,
    a_user_dict=a_user_dict,
    b_user_dict=b_user_dict,
    interact_dict=interact_dict,
    subscribe_dict=subscribe_dict,
    keyword_dict=keyword_dict,
    gender_vocab=gender_vocab,
    tag_vocab=tag_vocab,
    cate_vocab=cate_vocab,
    kw_vocab=kw_vocab,
    batch_size=STREAM_BATCH_SIZE
)
# 构建DataLoader（IterableDataset必须设置shuffle=False，此处省略shuffle参数）
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=0,  # 避免多进程问题
    pin_memory=True if torch.cuda.is_available() else False
)
print("RecommendIterableDataset流式数据集初始化完毕")






