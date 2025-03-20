import pandas as pd
import argparse
from pathlib import Path

def convert_to_parquet(csv_path, output_path=None):
    """将CSV文件转换为Parquet格式
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出Parquet文件路径，如果不指定则使用相同文件名
    """
    print(f"开始转换 {csv_path}...")
    
    # 1. 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV读取成功:")
        print(f"- 行数: {len(df)}")
        print(f"- 列数: {len(df.columns)}")
        print(f"- 列名: {df.columns.tolist()}")
    except Exception as e:
        print(f"CSV读取失败: {str(e)}")
        return False
    
    # 2. 确定输出路径
    if output_path is None:
        output_path = str(Path(csv_path).with_suffix('.parquet'))
    
    # 3. 保存为parquet格式
    try:
        df.to_parquet(output_path, index=False)
        print(f"\nParquet文件已保存: {output_path}")
    except Exception as e:
        print(f"Parquet保存失败: {str(e)}")
        return False
    
    # 4. 验证parquet文件
    try:
        df_verify = pd.read_parquet(output_path)
        print("\nParquet验证结果:")
        print(f"- 行数: {len(df_verify)}")
        print(f"- 列数: {len(df_verify.columns)}")
        print(f"- 列名: {df_verify.columns.tolist()}")
        
        # 验证数据是否一致
        if len(df) == len(df_verify) and all(df.columns == df_verify.columns):
            print("\n转换成功！数据完整性验证通过。")
            return True
        else:
            print("\n警告：转换后的数据与原数据不一致！")
            return False
    except Exception as e:
        print(f"\nParquet验证失败: {str(e)}")
        return False

if __name__ == '__main__':

    csv_path = 'yfcc15m_3long_3short_1raw_captions_url.csv'
    output_path = 'yfcc15m_3long_3short_1raw_captions_url.parquet'
    convert_to_parquet(csv_path, output_path)

