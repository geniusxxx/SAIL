import pandas as pd
import csv
from tqdm import tqdm

def analyze_csv_file(file_path):
    print("1. 基本信息:")
    # 使用原始方式计算行数
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"文件总行数: {total_lines}")
    
    # 使用pandas读取并分析
    print("\n2. Pandas读取分析:")
    df = pd.read_csv(file_path)
    print(f"Pandas读取行数: {len(df)}")
    print(f"重复行数量: {len(df) - len(df.drop_duplicates())}")
    
    # 检查URL列
    print("\n3. URL分析:")
    print(f"唯一URL数量: {df['Image Url'].nunique()}")
    print(f"重复URL数量: {len(df) - df['Image Url'].nunique()}")
    
    # 使用原始CSV读取进行对比
    print("\n4. 逐行分析:")
    urls = set()
    total_rows = 0
    duplicate_urls = 0
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        url_index = header.index('Image Url')
        
        for row in tqdm(reader):
            total_rows += 1
            url = row[url_index]
            if url in urls:
                duplicate_urls += 1
            urls.add(url)
    
    print(f"\n原始CSV统计:")
    print(f"总行数: {total_rows}")
    print(f"唯一URL数: {len(urls)}")
    print(f"重复URL数: {duplicate_urls}")
    
    # 对比pandas和原始读取
    print("\n5. 读取方式对比:")
    print(f"文件总行数: {total_lines}")
    print(f"Pandas读取行数: {len(df)}")
    print(f"原始读取行数: {total_rows}")
    print(f"差异: {total_lines - len(df)}")

if __name__ == "__main__":
    file_path = "cc3m_3long_3short_1raw_captions_url.csv"
    analyze_csv_file(file_path)