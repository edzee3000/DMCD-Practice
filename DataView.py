


def preview_file(file_path, n_lines=10, sep="\t"):
    """
    预览文件前n行，解析字段并打印
    :param file_path: 文件路径
    :param n_lines: 要预览的行数
    :param sep: 字段分隔符（数据集均为|）
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                # 去除换行符，按分隔符拆分字段
                line = line.strip()  # 去掉首尾空白/换行
                if not line:  # 跳过空行
                    print(f"第{i+1}行：空行")
                    continue
                fields = line.split(sep)
                print(f"第{i+1}行 | 内容：{fields}")
                print(line)
    except Exception as e:
        print(f"读取错误：{e}")

UserPath = "./Data/UserData"
n_line = 5
# 示例：预览训练集前5行
print(f"主数据集 subscribe_train 的前 {n_line} 行数据为:")
preview_file(f"{UserPath}/subscribe_train.txt", n_lines=n_line)
# 预览A类用户特征前5行
print(f"\nA类用户特征数据集 user_basic_info.txt 的前 {n_line} 行数据为:")
preview_file(f"{UserPath}/user_basic_info.txt", n_lines=n_line)
# 预览B类用户特征前5行
print(f"\nB类用户特征数据集 user_feature.txt 的前 {n_line} 行数据为:")
preview_file(f"{UserPath}/user_feature.txt", n_lines=n_line)
# 预览用户行为数据前5行
print(f"\n用户行为数据集 user_interaction.txt 的前 {n_line} 行数据为:")
preview_file(f"{UserPath}/user_interaction.txt", n_lines=n_line)
# 预览用户关注历史数据前5行
print(f"\n用户关注历史数据集 user_subscribe.txt 的前 {n_line} 行数据为:")
preview_file(f"{UserPath}/user_subscribe.txt", n_lines=n_line)
# 预览用户行为关键词数据前5行
print(f"\n用户行为关键词数据集 user_keyword.txt 的前 {n_line} 行数据为:")
preview_file(f"{UserPath}/user_keyword.txt", n_lines=n_line)
