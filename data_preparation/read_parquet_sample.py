import pandas as pd
import sys
import os
from pprint import pprint

def read_parquet_info(parquet_file, num_rows=5):
    """
    读取Parquet文件的基本信息和前几行数据
    
    参数:
        parquet_file: Parquet文件路径
        num_rows: 要显示的行数
    """
    if not os.path.exists(parquet_file):
        print(f"错误: 文件 {parquet_file} 不存在")
        return

    try:
        # 获取文件大小
        file_size = os.path.getsize(parquet_file) / (1024 * 1024)  # MB
        print(f"文件大小: {file_size:.2f} MB")
        
        # 读取Parquet文件的元数据
        metadata = pd.read_parquet(parquet_file, engine='pyarrow').info()
        print("\n文件元数据:")
        print(metadata)
        
        # 读取前几行数据
        print(f"\n前 {num_rows} 行数据:")
        df = pd.read_parquet(parquet_file, engine='pyarrow')
        
        # 显示列名
        print("\n列名:")
        print(df.columns.tolist())
        
        # 打印前几行数据
        print(f"\n数据样本 ({num_rows} 行):")
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.width', 1000)        # 增加显示宽度
        pd.set_option('display.max_colwidth', 30)   # 限制每列显示的字符数
        print(df.head(num_rows))
        
        # 如果有大型列（如向量），单独显示
        large_columns = [col for col in df.columns if isinstance(df[col].iloc[0], (list, tuple)) or 
                         (hasattr(df[col].iloc[0], '__len__') and len(df[col].iloc[0]) > 10)]
        
        if large_columns:
            print("\n大型列数据样本:")
            for col in large_columns:
                print(f"\n列 '{col}' 的第一个元素:")
                pprint(df[col].iloc[0])
                
        return df.head(num_rows)
        
    except Exception as e:
        print(f"读取Parquet文件时出错: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python read_parquet_sample.py <parquet_file_path> [num_rows]")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    read_parquet_info(parquet_file, num_rows) 