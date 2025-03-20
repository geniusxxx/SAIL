# import fsspec
# import pyarrow.csv as csv_pa
# import pyarrow as pa
# import math
# from multiprocessing.pool import ThreadPool
# import time

# class CSVReader:
#     def __init__(self, url_list, url_col, caption_col=None, save_additional_columns=None):
#         # 基本设置
#         self.url_col = url_col
#         self.caption_col = caption_col
#         self.save_additional_columns = save_additional_columns
#         self.number_sample_per_shard = 10000  # 设置一个合理的分片大小
        
#         # 获取文件系统和路径
#         self.fs, self.url_path = fsspec.core.url_to_fs(url_list)
#         self.input_files = [url_list]
        
#         # 设置列名列表
#         self.column_list = self.save_additional_columns if self.save_additional_columns is not None else []
#         if self.caption_col is not None:
#             self.column_list = self.column_list + ["caption"]
#         self.column_list = self.column_list + ["url"]

#     def read_csv(self):
#         """按照img2dataset的方式读取CSV文件"""
#         for i, input_file in enumerate(self.input_files):
#             print(f"Reading file {i + 1} of {len(self.input_files)}: {input_file}")
            
#             try:
#                 # 使用与img2dataset相同的读取方式
#                 with self.fs.open(input_file, encoding="utf-8", mode="rb") as file:
#                     df = csv_pa.read_csv(file)
                    
#                     # 重命名列
#                     column_names = df.column_names
#                     if self.caption_col is not None:
#                         column_names = [c if c != self.caption_col else "caption" for c in column_names]
#                     column_names = [c if c != self.url_col else "url" for c in column_names]
#                     df = df.rename_columns(column_names)
                    
#                     # 打印每个分片的信息
#                     number_samples = df.num_rows
#                     number_shards = math.ceil(number_samples / self.number_sample_per_shard)
                    
#                     print(f"Total rows: {number_samples}")
#                     print(f"Number of shards: {number_shards}")
                    
#                     # 检查每个分片
#                     for shard_id in range(number_shards):
#                         begin_shard = shard_id * self.number_sample_per_shard
#                         end_shard = min(number_samples, (1 + shard_id) * self.number_sample_per_shard)
#                         df_shard = df.slice(begin_shard, end_shard - begin_shard)
                        
#                         print(f"\nShard {shard_id + 1}:")
#                         print(f"Rows in shard: {df_shard.num_rows}")
#                         print(f"Columns in shard: {df_shard.num_columns}")
                        
#                         # 如果发现列数不正确，打印详细信息
#                         if df_shard.num_columns != 8:  # 期望的列数
#                             print("Found incorrect number of columns!")
#                             print("Row content:")
#                             print(df_shard.to_pandas().iloc[0].tolist())
                            
#             except Exception as e:
#                 print(f"Error reading file: {str(e)}")
#                 # 在这里添加打印错误行的代码
#                 error_text = b"These objectively existing and visible objects in the image provide insight into the history"
#                 print("\nTrying to locate the problematic line...")
#                 with self.fs.open(input_file, mode="rb") as f:  # 使用二进制模式打开
#                     for line_num, line in enumerate(f, 1):
#                         if error_text in line:
#                             print(f"\nFound problematic line (line {line_num}):")
#                             print("Line content:")
#                             print(line.decode('utf-8').strip())
#                             print("\nNumber of columns in this line:", len(line.decode('utf-8').strip().split(',')))
#                             break
#                 raise

# # 使用这个类来测试CSV文件
# reader = CSVReader(
#     url_list='cc3m_3long_3short_1raw_captions_url.csv',
#     url_col='Image Path',
#     caption_col='raw_caption',
#     save_additional_columns=['shortIB_captions', 'longIB_captions', 
#                            'shortSV_captions', 'longSV_captions',
#                            'shortLLA_captions', 'longLLA_captions']
# )

# reader.read_csv()

# import fsspec
# import pyarrow.csv as csv_pa
# import pyarrow as pa
# import math

# def test_csv_reading(file_path):
#     """严格按照img2dataset的方式测试CSV读取"""
    
#     # 获取文件系统（与img2dataset相同）
#     fs, url_path = fsspec.core.url_to_fs(file_path)
    
#     try:
#         # 完全按照img2dataset的方式打开和读取文件
#         with fs.open(file_path, encoding="utf-8", mode="rb") as file:
#             # 使用与img2dataset相同的读取方式
#             df = csv_pa.read_csv(file)
#             print("Successfully read the CSV file")
#             print(f"Number of columns: {df.num_columns}")
#             print(f"Number of rows: {df.num_rows}")
            
#     except Exception as e:
#         print(f"Error occurred during CSV reading: {str(e)}")
#         # 这个错误是pyarrow本身的错误，不是我们打印的
#         # pyarrow在遇到列数不匹配时会自动停止并报错
#         # 错误信息中的文本片段是pyarrow提供的

# # 测试读取
# test_csv_reading('cc3m_3long_3short_1raw_captions_url.csv')

# import fsspec
# import pyarrow.csv as csv_pa
# import pyarrow as pa
# import math

# def test_csv_reading(file_path):
#     """严格按照img2dataset的方式测试CSV读取"""
    
#     # 获取文件系统（与img2dataset相同）
#     fs, url_path = fsspec.core.url_to_fs(file_path)
    
#     try:
#         # 完全按照img2dataset的方式打开和读取文件
#         with fs.open(file_path, encoding="utf-8", mode="rb") as file:
#             df = csv_pa.read_csv(file)
#             print("Successfully read the CSV file")
#             print(f"Number of columns: {df.num_columns}")
#             print(f"Number of rows: {df.num_rows}")
            
#     except Exception as e:
#         print(f"Error occurred during CSV reading: {str(e)}")
#         print("\nAnalyzing the problematic line structure:")
        
#         # 使用更精确的匹配字符串
#         error_text = b"These objectively existing and visible objects in the image provide insight into the history"
        
#         with fs.open(file_path, mode="rb") as f:
#             for i, line in enumerate(f, 1):
#                 if error_text in line:
#                     line_str = line.decode('utf-8')
#                     print(f"\nLine {i} column count: {len(line_str.strip().split(','))}")
#                     print("Expected column count: 8")
#                     print("\nColumns after incorrect splitting:")
#                     for j, col in enumerate(line_str.strip().split(','), 1):
#                         print(f"Column {j}:")
#                         print(f"{col.strip()}")
#                         print("-" * 80)  # 添加分隔线使输出更清晰
#                     break

# # 测试读取
# test_csv_reading('cc3m_3long_3short_1raw_captions_url.csv')

# import pandas as pd

# def verify_csv_reading():
#     try:
#         # 读取CSV文件
#         df = pd.read_csv('cc3m_3long_3short_1raw_captions_url.csv')
        
#         # 1. 检查基本信息
#         print(f"列数: {len(df.columns)}")
#         print("列名:", df.columns.tolist())
#         print(f"总行数: {len(df)}")
        
#         # 2. 检查是否有异常的列数
#         # 如果每行都被正确解析，所有行的列数应该相同
#         print("\n数据结构是否正确:", len(df.columns) == 8)
        
#         return True, df
        
#     except Exception as e:
#         print(f"读取出错: {str(e)}")
#         return False, None

# success, df = verify_csv_reading()
# if success:
#     print("\nCSV文件读取正确！")

# import pandas as pd

# def verify_csv_reading():
#     try:
#         # 读取CSV文件
#         df = pd.read_csv('cc3m_3long_3short_1raw_captions_url.csv')
        
#         # 1. 基本信息检查
#         print(f"列数: {len(df.columns)}")
#         print("列名:", df.columns.tolist())
#         print(f"总行数: {len(df)}")
        
#         # 2. 数据结构检查
#         print("\n数据结构是否正确:", len(df.columns) == 8)
        
#         # 3. 内容合理性检查
#         print("\n数据内容验证:")
#         # 检查Image Path列是否都是URL格式
#         print("- Image Path列URL检查:", df['Image Path'].str.contains('http').all())
        
#         # 随机抽查几行，打印它们的内容
#         print("\n随机抽查3行数据:")
#         sample_rows = df.sample(n=3)
#         for idx, row in sample_rows.iterrows():
#             print(f"\n行号 {idx}:")
#             for col in df.columns:
#                 print(f"{col}: {row[col][:100]}...")  # 只打印前100个字符
        
#         return True, df
        
#     except Exception as e:
#         print(f"读取出错: {str(e)}")
#         return False, None

# success, df = verify_csv_reading()
# if success:
#     print("\nCSV文件读取正确！")

import pandas as pd
import pyarrow.csv as csv_pa
import fsspec

def compare_readers():
    file_path = 'cc3m_3long_3short_1raw_captions_url.csv'
    
    # 1. Pandas读取（不指定usecols，读取所有列）
    print("使用Pandas读取:")
    try:
        pd_df = pd.read_csv(file_path)
        print("Pandas成功读取:")
        print(f"- 行数: {len(pd_df)}")
        print(f"- 列数: {len(pd_df.columns)}")
        print("\n列名:")
        print(pd_df.columns.tolist())
        print("\n前3行内容:")
        print(pd_df.head(3))
    except Exception as e:
        print(f"Pandas读取错误: {str(e)}")
    
    # 2. PyArrow读取
    print("\n使用PyArrow读取:")
    try:
        fs, path = fsspec.core.url_to_fs(file_path)
        with fs.open(file_path, mode="rb") as file:
            pa_df = csv_pa.read_csv(file)
        print("PyArrow成功读取:")
        print(f"- 行数: {pa_df.num_rows}")
        print(f"- 列数: {pa_df.num_columns}")
        print("\n列名:")
        print(pa_df.column_names)
        print("\n前3行内容:")
        print(pa_df.slice(0, 3).to_pandas())
    except Exception as e:
        print(f"PyArrow读取错误: {str(e)}")

compare_readers()