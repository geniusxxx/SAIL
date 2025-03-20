import pandas as pd
import random
from tqdm import tqdm

def verify_data_matching(csv_file, num_samples=5, chunk_size=10000):
    """
    验证CSV文件中的图像URL、路径和描述是否正确匹配
    
    Args:
        csv_file: CSV文件路径
        num_samples: 要显示的随机样本数量
        chunk_size: 每次读取的行数
    """
    print(f"\n检查文件: {csv_file}")
    print("="*80)
    
    # 首先获取文件总行数
    total_rows = sum(1 for _ in open(csv_file)) - 1
    print(f"总行数: {total_rows}")
    
    # 获取列名
    df_header = pd.read_csv(csv_file, nrows=0)
    print(f"列名: {df_header.columns.tolist()}")
    
    # 随机选择要检查的行号
    sample_indices = sorted(random.sample(range(total_rows), num_samples))
    current_chunk_start = 0
    samples_found = []
    
    # 分块读取文件
    for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunk_size), 
                     desc="读取数据块", 
                     total=total_rows//chunk_size + 1):
        chunk_end = current_chunk_start + len(chunk)
        
        # 检查这个chunk是否包含我们要找的样本
        for idx in sample_indices:
            if current_chunk_start <= idx < chunk_end:
                row = chunk.iloc[idx - current_chunk_start]
                samples_found.append((idx, row))
        
        current_chunk_start = chunk_end
        
        # 如果找到了所有样本，就可以停止了
        if len(samples_found) == len(sample_indices):
            break
    
    # 显示找到的样本
    print("\n随机样本验证:")
    for i, row in samples_found:
        print("\n样本 #", i)
        print("-"*40)
        print(f"Image Url: {row['Image Url']}")
        print(f"Image Path: {row['Image Path']}")
        print(f"Raw Caption: {row['raw_caption']}")
        
        # 验证路径格式是否正确
        dataset_name = csv_file.split('_')[0]
        folder_num = int(row['Image Path'].split('/')[2])
        img_num = int(row['Image Path'].split('/')[3].split('.')[0])
        expected_index = folder_num * 10000 + img_num
        
        print(f"\n路径验证:")
        print(f"数据集名称: {dataset_name}")
        print(f"文件夹编号: {folder_num}")
        print(f"图片编号: {img_num}")
        print(f"推算的索引: {expected_index}")
        print(f"实际行号: {i}")
        
        if abs(expected_index - i) > 1:  # 允许1的误差
            print("⚠️ 警告：索引可能不匹配！")
        else:
            print("✓ 索引匹配正确")

# 验证所有数据集
files = [
    'yfcc15m_3long_3short_1raw_captions_url.csv',
    'cc3m_3long_3short_1raw_captions_url.csv',
    'cc12m_3long_3short_1raw_captions_url.csv'
]

for file in files:
    try:
        verify_data_matching(file)
    except Exception as e:
        print(f"处理文件 {file} 时出错: {str(e)}")