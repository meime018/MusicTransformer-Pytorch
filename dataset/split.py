import os
import json
import pickle
import random
from shutil import copyfile

def get_all_json_files(root_dir):
    json_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(subdir, file))
    return json_files

maestro_dir = "Maestro_tokens_no_bpe"  # 替换为您的MAESTRO目录路径
json_files = get_all_json_files(maestro_dir)

random.shuffle(json_files)
total_files = len(json_files)
train_files = json_files[:int(total_files * 0.8)]
test_files = json_files[int(total_files * 0.8):int(total_files * 0.9)]
val_files = json_files[int(total_files * 0.9):]

os.makedirs("e_piano/train", exist_ok=True)
os.makedirs("e_piano/test", exist_ok=True)
os.makedirs("e_piano/val", exist_ok=True)

def copy_and_convert_json_to_pickle(src_files, target_dir):
    for file_path in src_files:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # 提取 'ids' 键下的列表
        ids_data = data['ids'][0]  # 假设 'ids' 总是包含一个列表

        base_name = os.path.basename(file_path)
        pickle_file_name = os.path.splitext(base_name)[0] + '.pkl'
        target_path = os.path.join(target_dir, pickle_file_name)

        # 将提取的数据保存为Pickle文件
        with open(target_path, 'wb') as pickle_file:
            pickle.dump(ids_data, pickle_file)

copy_and_convert_json_to_pickle(train_files, "e_piano/train")
copy_and_convert_json_to_pickle(test_files, "e_piano/test")
copy_and_convert_json_to_pickle(val_files, "e_piano/val")
