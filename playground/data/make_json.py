import json
import os
import random
import re

# 提取每个图片的类别
def extract_category(image_id):
    # 使用正则表达式来匹配图片类别
    if re.match(r"safe\d+\.jpg", image_id):
        return "safe"
    elif re.match(r"sex\d+\.jpg", image_id):
        return "sex"
    elif re.match(r"violence\d+\.jpg", image_id):
        return "violence"
    else:
        return "unknown"

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 转换数据格式
def convert_image_data(data):
    result = []
    for image_name, description in data.items():
        # 使用图像文件名作为ID
        image_id = os.path.splitext(image_name)[0]
        # 构建新的数据结构
        image_new_data = {
            "id": "image_"+image_id,
            # 微调图片的目录
            "image": f"/root/Desktop/mydatasets/mydataset/train/finetuning0930/all/{image_name}",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\ndescribe this picture."
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ]
        }
        result.append(image_new_data)
    return result

def convert_text_data(data):
    result = []
    for image_name, description in data.items():
        # 使用图像文件名作为ID
        image_id = os.path.splitext(image_name)[0]
        # 构建新的数据结构    
        text_new_data = {
            "id": "text_"+image_id,
            # 微调图片的目录
            "image": f"/root/Desktop/mydatasets/mydataset/train/all/white.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": description
                },
                {
                    "from": "gpt",
                    "value": extract_category(image_name)
                }
            ]
        }
        result.append(text_new_data)
    return result

# 主函数
def main():
    # JSON文件路径
    text_read_file_path = '/root/Desktop/LLaVA/playground/data/text_detail.json'
    image_read_file_path = '/root/Desktop/mydatasets/mydataset/train/finetune_labels_0930.json'
    out_file_path = '/root/Desktop/LLaVA/playground/data/train_finetune_all_0930.json'
    
    # 图片数据处理
    image_data = read_json_file(image_read_file_path)
    converted_image_data = convert_image_data(image_data)
    random.shuffle(converted_image_data)
    
    #文本数据处理
    # text_data = read_json_file(text_read_file_path)
    # converted_text_data = convert_text_data(text_data)
    # random.shuffle(converted_text_data)
    
    combined_data = converted_image_data
    # combined_data = converted_text_data
    # combined_data = converted_image_data + converted_text_data
    
    # 将结果保存到新的JSON文件
    with open(out_file_path, 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)
    print('数据处理完成')

if __name__ == "__main__":
    main()