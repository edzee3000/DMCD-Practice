from utils import *
from Dataset import *
from Models import *
from DataProcess import *


# 初始化模型
embed_config = [
    (len(gender_vocab), 4),    # A性别嵌入
    (len(tag_vocab), 8),       # A标签嵌入
    (len(cate_vocab), 8),      # B一级分类嵌入
    (len(cate_vocab), 8),      # B二级分类嵌入
    (len(cate_vocab), 8),      # B三级分类嵌入
    (len(cate_vocab), 8),      # B四级分类嵌入
    (len(kw_vocab), 8),        # B关键词嵌入
    (len(kw_vocab), 8)         # A关键词嵌入
]

model = WideDeep(
    wide_dim=8,
    deep_numeric_dim=8,
    embed_config=embed_config
).to(device)


# 损失函数和优化器
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 混合精度训练（可选，加速GPU训练）
scaler = GradScaler() if torch.cuda.is_available() else None


# ====================== 6. 训练模型（适配大数据集） ======================
def train_large_dataset(model, train_txt_path):
    """训练2G大训练集：流式读取，分批次训练"""
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        sample_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        start_time = time.time()  # 批次计时起始

        for wide_x, deep_x, labels in pbar:
            # 重置下一批次计时
            batch_start_time = time.time()

            # 数据移到设备
            wide_x = wide_x.to(device, non_blocking=True)
            deep_x = deep_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 前向传播（混合精度）
            if scaler is not None:
                with autocast():
                    preds = model(wide_x, deep_x)
                    loss = criterion(preds, labels)
            else:
                preds = model(wide_x, deep_x)
                loss = criterion(preds, labels)
            
            # 反向传播
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # 统计损失
            batch_loss = loss.item() * wide_x.shape[0]
            total_loss += batch_loss
            sample_count += wide_x.shape[0]  # 加上当前批次样本数
            avg_loss = total_loss / sample_count

            # # 统计速率
            # batch_end_time = time.time()
            # batch_elapsed = batch_end_time - batch_start_time  # 当前批次耗时
            # batch_samples = wide_x.shape[0]  # 当前批次样本数
            # # 瞬时速率：当前批次样本数 / 当前批次耗时
            # batch_samples_per_sec = batch_samples / batch_elapsed if batch_elapsed > 1e-6 else 0.0
            # # 累计速率计算
            # elapsed_time = batch_end_time - start_time
            # total_samples_per_sec = sample_count / elapsed_time if elapsed_time > 1e-6 else 0.0
            
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "sample_count": f"{sample_count}",
                # "total_samples/s": f"{total_samples_per_sec:.2f}",  # 累计平均
                # "batch_samples/s": f"{batch_samples_per_sec:.2f}"   # 批次瞬时
            })
        
        # 保存每个epoch的模型
        model_path = f"./WideDeep_Epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\nEpoch {epoch+1} 完成，模型已保存到 {model_path}")


# ====================== 7. 启动训练 ======================
if __name__ == "__main__":
    print("\n开始训练Wide&Deep模型...")
    train_large_dataset(model, f"{TRAIN_DATA_PATH}")
    print("\n训练全部完成！")