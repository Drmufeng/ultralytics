# label_generator.py

import os
import json


def combine_js(file_path, out_path):
    data = {}

    for filename in os.listdir(file_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(file_path, filename)

            with open(json_file_path, 'r') as json_file:
                json_data = json.load(json_file)
                # 检查空数据
                has_skeleton = any(
                    any(skeleton['pose'] or skeleton['score'] for skeleton in frame['skeleton']) for frame in
                    json_data['data'])

                entry = {
                    "has_skeleton": has_skeleton,
                    "label": json_data["label"],
                    "label_index": json_data["label_index"]
                }

                data[filename.split('.')[0]] = entry

    with open(out_path, 'w') as output_json_file:
        json.dump(data, output_json_file, indent=4)


# 调用函数，传入文件夹路径和输出文件名
combine_js("runs/JSON/val", "runs/yoloJSON/vallabel.json")
combine_js("runs/JSON/train", "runs/yoloJSON/trainlabel.json")