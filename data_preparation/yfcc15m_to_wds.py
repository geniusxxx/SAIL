import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import tarfile
import tempfile
import io
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys
import time
import clip
import base64
import shutil


# 尝试导入CLIP tokenizer
try:
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
except ImportError:
    print("尝试安装CLIP...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

def decode_tokens(tokens, tokenizer):
    """将token向量解码为原始文本"""
    # print(f"原始tokens类型: {type(tokens)}, 值: {tokens}")
    
    # 统一处理成列表类型
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    elif isinstance(tokens, dict) and 'list' in tokens:
        tokens = tokens['list']
    elif not isinstance(tokens, list):
        try:
            tokens = list(tokens)
        except:
            # print(f"无法将tokens转换为列表: {tokens}")
            return ""
    
    # 确保tokens是整数列表
    int_tokens = []
    for t in tokens:
        if isinstance(t, (int, np.integer)):
            int_tokens.append(int(t))
        elif isinstance(t, (float, np.floating)) and t.is_integer():
            int_tokens.append(int(t))
    
    # 过滤掉特殊tokens (0: padding, 1: 开始, 2: 结束)
    filtered_tokens = [t for t in int_tokens if t > 2]
    
    # print(f"过滤后tokens: {filtered_tokens[:10]}...共{len(filtered_tokens)}个")
    
    if len(filtered_tokens) > 0:
        try:
            # 使用CLIP tokenizer将tokens解码为文本
            text = tokenizer.decode(filtered_tokens)
            
            # 去掉特殊标记
            text = remove_special_tokens(text)
            
            # print(f"解码后文本: {text[:100]}...")
            return text
        except Exception as e:
            # print(f"使用tokenizer解码错误: {str(e)}")
            # 尝试手动解码最常见的tokens
            try:
                text = manual_decode_tokens(filtered_tokens)
                text = remove_special_tokens(text)
                return text
            except Exception as e2:
                # print(f"手动解码也失败: {str(e2)}")
                return ""
    return ""

def remove_special_tokens(text):
    """移除文本中的特殊标记"""
    text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
    # 去掉首尾空格
    return text.strip()

def manual_decode_tokens(tokens):
    """
    手动将tokens解码为文本 (简化版)
    当CLIP tokenizer不可用或失败时使用
    """
    try:
        # 打印tokens范围，帮助诊断
        token_min = min(tokens) if tokens else 0
        token_max = max(tokens) if tokens else 0
        # print(f"手动解码tokens范围: {token_min}-{token_max}")
        
        # 使用utf-8编码直接转换tokens
        # 这是一个简化方法，CLIP实际使用更复杂的tokenization
        byte_tokens = bytes(tokens)
        text = byte_tokens.decode('utf-8', errors='replace')
        return text
    except Exception as e:
        # print(f"手动解码出错: {str(e)}")
        return "TOKEN_DECODE_FAILED"

# 自定义JSON编码器，处理非JSON可序列化类型
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def process_batch(df_batch, output_tar, tokenizer, start_idx, samples_per_tar=1000):
    """处理一批数据并将其写入tar文件"""
    temp_dir = tempfile.mkdtemp()
    files_created = []
    
    try:
        # 获取数据框的列名
        columns = df_batch.columns.tolist()
        # print(f"处理数据列: {columns}")
        
        # 检查一下第一行数据的结构，帮助调试
        # if not df_batch.empty:
        #     print("\n===== 第一行数据结构 =====")
        #     for col in columns:
        #         print(f"列 '{col}' 类型: {type(df_batch[col].iloc[0])}")
        #         # 尝试打印前100个字符看看数据样式
        #         try:
        #             print(f"样本: {str(df_batch[col].iloc[0])[:100]}...")
        #         except:
        #             print("无法打印样本")
        #     print("===========================\n")
        
        for i, row in enumerate(df_batch.itertuples()):
            idx = start_idx + i
            prefix = f"{idx:08d}"
            
            # print(f"\n处理第 {i} 条记录 (全局索引: {idx})")
            
            # 1. 保存图像 - 从原始数据中提取并保存为文件
            if hasattr(row, 'images'):  # 注意：列名是'images'而不是'image'
                # print(f"提取图像数据，类型: {type(row.images)}")
                image_data = row.images
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image_data = image_data['bytes']
                
                image_path = os.path.join(temp_dir, f"{prefix}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                files_created.append(image_path)
                # print(f"已保存图像到: {image_path}")
            
            # 2. 创建仅包含文本信息的JSON
            json_data = {}
            
            # 只添加文本信息，不包括图像数据
            if hasattr(row, 'texts'):
                # print(f"提取texts数据，类型: {type(row.texts)}")
                try:
                    # 使用索引1是因为row[0]是行索引
                    col_idx = columns.index('texts') + 1
                    tokens = row[col_idx]
                    # print(f"使用索引获取tokens: {type(tokens)}")
                except Exception as e:
                    # print(f"使用索引获取tokens失败: {str(e)}")
                    tokens = row.texts
                
                text = decode_tokens(tokens, tokenizer)
                json_data['caption'] = text
                # print(f"解码后caption: {text[:30]}...")
            
            # 保存JSON
            json_path = os.path.join(temp_dir, f"{prefix}.json")
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False)
                files_created.append(json_path)
            except Exception as e:
                # print(f"保存JSON文件失败: {e}")
                # 尝试使用更简单的JSON结构
                simple_json = {
                    'caption': json_data.get('caption', '')
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_json, f, ensure_ascii=False)
                files_created.append(json_path)
            
            # 调试：打印前3个样本的JSON内容
            # if i < 3:
            #     with open(json_path, 'r', encoding='utf-8') as f:
            #         print(f"样本 {i} JSON内容: {f.read()}")
        
        # 将所有文件添加到tar
        for file_path in files_created:
            arcname = os.path.basename(file_path)
            output_tar.add(file_path, arcname=arcname)
            
    finally:
        # 更健壮的临时文件清理
        try:
            for file_path in files_created:
                if os.path.exists(file_path):
                    os.remove(file_path)
            # 使用shutil递归删除目录，即使不为空
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"清理临时文件时出错: {e}")

def convert_parquet_to_wds(parquet_path, output_dir, samples_per_tar=1000, num_samples=None):
    """将parquet文件转换为WebDataset格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化tokenizer
    print("初始化CLIP tokenizer...")
    tokenizer = _Tokenizer()
    
    # 获取parquet文件的行数
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Parquet文件共有 {total_rows} 行")
    
    if num_samples is not None and num_samples < total_rows:
        total_rows = num_samples
        print(f"将处理前 {num_samples} 行")
    
    # 计算需要多少个tar文件
    num_tars = (total_rows + samples_per_tar - 1) // samples_per_tar
    print(f"将创建 {num_tars} 个tar文件，每个文件包含最多 {samples_per_tar} 个样本")
    
    start_time = time.time()
    processed_rows = 0
    
    # 按批次读取并处理
    for tar_idx in tqdm(range(num_tars), desc="创建tar文件"):
        # 计算当前批次的起始位置
        start_row = tar_idx * samples_per_tar
        if start_row >= total_rows:
            break
            
        # 计算当前批次的结束位置
        end_row = min(start_row + samples_per_tar, total_rows)
        current_batch_size = end_row - start_row
        
        # 创建tar文件
        tar_filename = os.path.join(output_dir, f"{tar_idx:06d}.tar")
        print(f"\n创建tar文件 {tar_idx+1}/{num_tars}: {tar_filename} (行范围: {start_row}-{end_row-1}, 共{current_batch_size}行)")
        
        with tarfile.open(tar_filename, 'w') as tar:
            # 尝试使用不同的方法读取parquet批次
            try:
                # print(f"尝试读取批次 {tar_idx}, 行范围: {start_row}-{end_row-1}")
                # print("读取方法1: 尝试使用pyarrow直接读取指定的行范围")
                table = pq.read_table(parquet_path, use_threads=True)
                batch_table = table.slice(start_row, current_batch_size)  # 使用实际批次大小
                batch_df = batch_table.to_pandas()
                
                if batch_df.empty:
                    raise Exception("直接读取得到空DataFrame")
                    
            except Exception as e1:
                print(f"直接读取失败: {str(e1)}")
                try:
                    # 2. 尝试使用普通方式读取整个文件然后切片
                    # print(f"读取方法2: 使用pandas读取全部并切片")
                    df = pd.read_parquet(parquet_path, engine='pyarrow')
                    batch_df = df.iloc[start_row:end_row].copy()  # 使用实际批次范围
                    
                except Exception as e2:
                    print(f"读取并切片失败: {str(e2)}")
                    print("无法读取数据，跳过此批次")
                    continue
            
            # 如果批次为空，则跳过
            if batch_df.empty:
                print(f"批次 {tar_idx} 为空，跳过")
                continue
            
            # 调试：打印列名和数据类型
            # print(f"DataFrame列: {batch_df.columns.tolist()}")
            # print(f"数据类型: {batch_df.dtypes}")
            print(f"读取了 {len(batch_df)} 行数据")
            
            # 处理批次并写入tar
            try:
                process_batch(batch_df, tar, tokenizer, start_row, samples_per_tar)
                
                processed_rows += len(batch_df)
                
                # 显示进度
                elapsed_time = time.time() - start_time
                samples_per_sec = processed_rows / elapsed_time if elapsed_time > 0 else 0
                eta = (total_rows - processed_rows) / samples_per_sec if samples_per_sec > 0 else 0
                
                print(f"处理进度: {processed_rows}/{total_rows} 样本 "
                      f"({processed_rows/total_rows*100:.2f}%) | "
                      f"速度: {samples_per_sec:.2f} 样本/秒 | "
                      f"已用时间: {elapsed_time:.2f}秒 | "
                      f"预计剩余时间: {eta:.2f}秒")
            except Exception as e:
                print(f"处理批次 {tar_idx} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    total_time = time.time() - start_time
    print(f"\n转换完成! 总共处理了 {processed_rows} 个样本，耗时 {total_time:.2f} 秒")
    print(f"WebDataset文件保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="将yfcc15m parquet文件转换为WebDataset格式")
    parser.add_argument("--input", type=str, required=True, help="输入的parquet文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出WebDataset文件的目录")
    parser.add_argument("--samples_per_tar", type=int, default=1000, help="每个tar文件中的样本数量")
    parser.add_argument("--num_samples", type=int, default=None, help="要处理的样本总数 (None表示处理全部)")
    parser.add_argument("--debug", action="store_true", help="调试模式，详细输出处理过程")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    convert_parquet_to_wds(args.input, args.output, args.samples_per_tar, args.num_samples)

if __name__ == "__main__":
    main() 