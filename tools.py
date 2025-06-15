import os
import shutil
import random
from tqdm import tqdm
from pathlib import Path

def build_new_train(train_dir, val_dir, output_dir, val_ratio=0.3):
    os.makedirs(output_dir, exist_ok=True)

    for category in tqdm(os.listdir(train_dir)):
        train_cat_path = Path(train_dir) / category
        val_cat_path = Path(val_dir) / category
        output_cat_path = Path(output_dir) / category

        if not train_cat_path.is_dir():
            continue

        os.makedirs(output_cat_path, exist_ok=True)

        # 获取当前类别的所有样本
        train_files = list(train_cat_path.glob("*"))
        val_files = list(val_cat_path.glob("*")) if val_cat_path.exists() else []

        total_needed = len(train_files)
        num_from_val = int(total_needed * val_ratio)
        num_from_train = total_needed - num_from_val

        # 抽样
        sampled_val = random.sample(val_files, min(num_from_val, len(val_files)))
        remaining_needed = total_needed - len(sampled_val)
        sampled_train = random.sample(train_files, min(remaining_needed, len(train_files)))

        # 如果某个集合样本不够，再次随机补齐
        while len(sampled_val) < num_from_val and len(val_files) > 0:
            sampled_val.append(random.choice(val_files))
        while len(sampled_train) < remaining_needed and len(train_files) > 0:
            sampled_train.append(random.choice(train_files))

        # 复制文件
        for f in sampled_val + sampled_train:
            shutil.copy(f, output_cat_path / f.name)

        #print(f"[{category}] 构建完成: 从 val 取 {len(sampled_val)}，从 train 取 {len(sampled_train)}，共 {total_needed} 张")

# 示例调用
build_new_train(
    train_dir="../datasets/PlantCLEF500/train",
    val_dir="../datasets/PlantCLEF500/val",
    output_dir="../datasets/PlantCLEF50003/train",
    val_ratio=0.03  # 比如30%来自val
)
