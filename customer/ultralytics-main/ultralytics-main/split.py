import os
import json
import random
import shutil


def split_and_copy_json_files(data_folder, train_folder, test_folder, split_ratio=0.8):
    json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    random.shuffle(json_files)

    # Calculate the number of files for the training set
    num_train_files = int(len(json_files) * split_ratio)
    train_files = json_files[:num_train_files]
    test_files = json_files[num_train_files:]

    # Create train and test folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Copy the selected JSON files to train folder
    for file in train_files:
        source_path = os.path.join(data_folder, file)
        dest_path = os.path.join(train_folder, file)
        shutil.copy(source_path, dest_path)

    # Copy the selected JSON files to test folder
    for file in test_files:
        source_path = os.path.join(data_folder, file)
        dest_path = os.path.join(test_folder, file)
        shutil.copy(source_path, dest_path)


labels = ['BaseballPitch', 'Basketball', 'Bowling', 'CricketBowling']  # 'jogging','running','walking'
# 设置文件夹路径和分割比例

for i in labels:
    data_folder = 'runs/JSON/' + i
    train_folder = "runs/JSON/train"
    test_folder = "runs/JSON/val"
    split_ratio = 0.8  # 80% for training, 20% for testing

    split_and_copy_json_files(data_folder, train_folder, test_folder, split_ratio)