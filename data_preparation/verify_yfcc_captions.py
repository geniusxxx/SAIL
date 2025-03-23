#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：比较生成的YFCC15M WDS数据集中的caption与parquet文件中的raw_caption是否匹配
用于验证"原始文本应该就是新数据集的raw_caption"的假设
"""

import os
import json
import pandas as pd
import pyarrow.parquet as pq
import tarfile
import tempfile
import random
import argparse
from tabulate import tabulate
from clip.simple_tokenizer import SimpleTokenizer


def extract_jsons_from_tar(tar_path, num_samples=5):
    """从tar文件中提取几个JSON文件，优先选择索引靠前的样本"""
    results = []
    temp_dir = tempfile.mkdtemp(prefix="yfcc_verify_")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # 获取所有JSON文件
            json_members = [m for m in tar.getmembers() if m.name.endswith('.json')]
            if not json_members:
                print(f"警告：在{tar_path}中未找到JSON文件")
                return []
                
            # 排序并选择前几个
            json_members.sort(key=lambda x: x.name)
            selected = json_members[:num_samples]
            print(f"从{tar_path}中提取{len(selected)}个JSON文件")
            
            for member in selected:
                tar.extract(member, path=temp_dir)
                json_path = os.path.join(temp_dir, member.name)
                
                # 同时提取对应的jpg文件(用于辅助验证)
                image_name = member.name.replace('.json', '.jpg')
                try:
                    image_member = tar.getmember(image_name)
                    tar.extract(image_member, path=temp_dir)
                    image_path = os.path.join(temp_dir, image_name)
                except KeyError:
                    image_path = None
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        prefix = os.path.splitext(member.name)[0]
                        results.append({
                            'file': member.name,
                            'prefix': prefix,
                            'image_path': image_path,
                            'json_data': data
                        })
                    except json.JSONDecodeError:
                        print(f"错误：无法解析JSON文件 {json_path}")
    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def extract_sample_from_parquet(parquet_path, num_samples=5):
    """从parquet文件中提取样本数据"""
    try:
        # 使用pyarrow读取前N行
        table = pq.read_table(parquet_path)
        df = table.slice(0, num_samples).to_pandas()
        print(f"从{parquet_path}中读取了{len(df)}行数据")
        
        # 返回列名
        columns = df.columns.tolist()
        print(f"Parquet文件列名: {columns}")
        
        # 检查是否包含captions相关列
        caption_columns = [col for col in columns if 'caption' in col.lower()]
        if not caption_columns:
            print(f"警告: 未在parquet文件中找到caption相关列")
        else:
            print(f"找到caption相关列: {caption_columns}")
        
        return df
    except Exception as e:
        print(f"读取parquet文件出错: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="验证YFCC15M WDS数据集中的caption与parquet文件中的raw_caption是否匹配")
    parser.add_argument("--tar_path", type=str, default="/mnt/e/Datasets/yfcc15m_wds/000000.tar", help="生成的YFCC15M WDS tar文件路径")
    parser.add_argument("--parquet_path", type=str, default="/home/xuboyu/Projects/CLIP/test_mobileclip/SAIL/data_preparation/yfcc15m_3long_3short_1raw_captions_url.parquet", help="原始YFCC15M parquet文件路径")
    parser.add_argument("--samples", type=int, default=5, help="要检查的样本数量")
    
    args = parser.parse_args()
    
    # 1. 提取WDS中的JSON文件
    print(f"\n1. 从WDS文件中提取JSON样本...")
    wds_samples = extract_jsons_from_tar(args.tar_path, args.samples)
    if not wds_samples:
        print("错误: 未能从WDS文件中提取样本")
        return
    
    # 2. 读取parquet中的样本
    print(f"\n2. 从parquet文件中读取样本...")
    parquet_samples = extract_sample_from_parquet(args.parquet_path, args.samples)
    if parquet_samples is None:
        print("错误: 未能从parquet文件中读取样本")
        return
    
    # 3. 比较WDS caption与parquet中的raw_caption
    print(f"\n3. 比较caption内容...")
    
    # 打印WDS中的caption
    print("\nWDS数据集中的caption:")
    wds_captions = []
    for idx, sample in enumerate(wds_samples):
        caption = sample['json_data'].get('caption', '')
        wds_captions.append(caption)
        print(f"  样本{idx+1}: {caption[:100]}..." if len(caption) > 100 else f"  样本{idx+1}: {caption}")
    
    # 尝试从parquet中获取原始caption
    print("\nParquet数据集中的caption:")
    parquet_captions = []
    
    # 检查parquet文件的列名以确定正确的caption列
    if 'raw_caption' in parquet_samples.columns:
        caption_column = 'raw_caption'
    elif 'caption' in parquet_samples.columns:
        caption_column = 'caption'
    else:
        # 尝试查找caption相关列
        caption_columns = [col for col in parquet_samples.columns if 'caption' in col.lower()]
        caption_column = caption_columns[0] if caption_columns else None
    
    if caption_column:
        for idx, row in parquet_samples.iterrows():
            caption = row[caption_column]
            parquet_captions.append(caption)
            print(f"  样本{idx+1}: {caption[:100]}..." if len(str(caption)) > 100 else f"  样本{idx+1}: {caption}")
    else:
        print("  未找到caption相关列")
    
    # 4. 结论
    print("\n4. 验证结论:")
    
    if not wds_captions or not parquet_captions:
        print("  ❌ 无法进行验证，缺少数据")
        return
    
    # 比较两组caption是否内容相似
    matches = 0
    partial_matches = 0
    
    comparison_table = []
    for i in range(min(len(wds_captions), len(parquet_captions))):
        wds_cap = wds_captions[i] if i < len(wds_captions) else ''
        parq_cap = parquet_captions[i] if i < len(parquet_captions) else ''
        
        # 简单的相似度检查
        if wds_cap == parq_cap:
            matches += 1
            status = "✓ 完全匹配"
        elif wds_cap in parq_cap or parq_cap in wds_cap:
            partial_matches += 1
            status = "~ 部分匹配"
        else:
            # 计算词汇重叠度
            wds_words = set(wds_cap.lower().split())
            parq_words = set(parq_cap.lower().split())
            overlap = len(wds_words.intersection(parq_words))
            
            if overlap > 0:
                partial_matches += 1
                status = f"~ 词汇重叠 ({overlap}词)"
            else:
                status = "✗ 不匹配"
        
        comparison_table.append({
            "序号": i+1,
            "WDS Caption": (wds_cap[:60] + "...") if len(wds_cap) > 60 else wds_cap,
            "Parquet Caption": (parq_cap[:60] + "...") if len(parq_cap) > 60 else parq_cap,
            "状态": status
        })
    
    # 打印比较表格
    print("\n比较结果:")
    print(tabulate(comparison_table, headers="keys", tablefmt="grid"))
    
    # 统计匹配率
    total_compared = min(len(wds_captions), len(parquet_captions))
    if total_compared > 0:
        match_rate = matches / total_compared
        partial_match_rate = partial_matches / total_compared
        
        print(f"\n匹配统计:")
        print(f"  完全匹配: {matches}/{total_compared} ({match_rate:.1%})")
        print(f"  部分匹配: {partial_matches}/{total_compared} ({partial_match_rate:.1%})")
        
        if match_rate > 0.7 or (match_rate + partial_match_rate) > 0.9:
            print("\n✅ 结论: WDS数据集的caption与parquet文件的raw_caption具有高度一致性")
            print("   验证了假设'原始文本应该就是新数据集的raw_captio'")
        elif match_rate + partial_match_rate > 0.5:
            print("\n⚠️ 结论: WDS数据集的caption与parquet文件的raw_caption有一定相关性，但不完全一致")
        else:
            print("\n❌ 结论: WDS数据集的caption与parquet文件的raw_caption差异较大")


if __name__ == "__main__":
    # 处理可能的ImportError
    try:
        from clip.simple_tokenizer import SimpleTokenizer
    except ImportError:
        import sys
        import subprocess
        print("安装CLIP...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
        from clip.simple_tokenizer import SimpleTokenizer
    
    main() 