import pandas as pd
import csv
from tqdm import tqdm

def find_missing_rows(file_path, num_samples=5):
    print("分析未被加载的行...")
    
    # 记录所有行号
    all_line_numbers = set(range(1, sum(1 for _ in open(file_path))))
    
    # 记录pandas成功读取的行号
    loaded_line_numbers = set()
    df = pd.read_csv(file_path)
    
    # 使用csv模块读取并记录成功的行号
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过header
        
        for i, row in enumerate(reader, 1):
            if len(row) == len(header):  # 如果行能被正确解析
                loaded_line_numbers.add(i)
    
    # 找出未被加载的行号
    missing_line_numbers = all_line_numbers - loaded_line_numbers
    print(f"\n总行数: {len(all_line_numbers)}")
    print(f"成功加载行数: {len(loaded_line_numbers)}")
    print(f"未加载行数: {len(missing_line_numbers)}")
    
    # 显示一些未加载的行的内容
    print(f"\n显示{num_samples}个未加载的行样本:")
    samples = sorted(list(missing_line_numbers))[:num_samples]
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line_num in samples:
            print(f"\n行号 {line_num}:")
            print("内容:", lines[line_num])
            print("长度:", len(lines[line_num]))
            print("字符预览:", repr(lines[line_num][:200]))
            
            # 尝试解析这一行
            try:
                parsed = list(csv.reader([lines[line_num]]))[0]
                print("列数:", len(parsed))
                print("解析结果:", parsed[:3], "...")
            except Exception as e:
                print("解析错误:", str(e))

if __name__ == "__main__":
    file_path = "cc3m_3long_3short_1raw_captions_url.csv"
    find_missing_rows(file_path) 