#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：比较转换后的JSON文件中的captions与原始Parquet/CSV中的数据
"""

import os
import json
import pandas as pd
import tarfile
import tempfile
import random
import argparse
from tabulate import tabulate


def extract_random_jsons_from_tar(tar_path, num_samples=3):
    """从tar文件中随机提取几个JSON文件"""
    results = []
    temp_dir = tempfile.mkdtemp(prefix="caption_verify_")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # 获取所有JSON文件
            json_members = [m for m in tar.getmembers() if m.name.endswith('.json')]
            if not json_members:
                print(f"警告：在{tar_path}中未找到JSON文件")
                return []
                
            # 随机选择几个
            selected = random.sample(json_members, min(num_samples, len(json_members)))
            print(f"从{tar_path}中随机抽取{len(selected)}个JSON文件")
            
            for member in selected:
                tar.extract(member, path=temp_dir)
                json_path = os.path.join(temp_dir, member.name)
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        url = data.get('url', '')
                        if url:
                            results.append({
                                'file': member.name,
                                'url': url,
                                'json_data': data
                            })
                    except json.JSONDecodeError:
                        print(f"错误：无法解析JSON文件 {json_path}")
    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def find_in_parquet(parquet_path, url):
    """在Parquet文件中查找指定URL的数据行"""
    df = pd.read_parquet(parquet_path)
    # 查找URL匹配的行
    match = df[df['Image Path'] == url]
    if len(match) == 0:
        return None
    return match.iloc[0].to_dict()


def find_in_csv(csv_path, url):
    """在CSV文件中查找指定URL的数据行"""
    df = pd.read_csv(csv_path)
    # 查找URL匹配的行
    match = df[df['Image Path'] == url]
    if len(match) == 0:
        return None
    return match.iloc[0].to_dict()


def compare_captions(json_data, source_data):
    """比较JSON和原始数据中的caption内容"""
    result = {"字段": [], "JSON内容": [], "原始数据": [], "是否一致": []}
    
    caption_fields = [
        'raw_caption', 
        'shortIB_captions', 
        'longIB_captions', 
        'shortSV_captions', 
        'longSV_captions', 
        'shortLLA_captions', 
        'longLLA_captions'
    ]
    
    for field in caption_fields:
        # JSON中应该是数组格式
        json_value = json_data.get(field, [''])[0] if field in json_data else ""
        
        # CSV/Parquet中应该是单一字符串
        source_value = source_data.get(field, "") if field in source_data else ""
        if field == "raw_caption" and field not in source_data:
            # 处理不同的字段名
            source_value = source_data.get("caption", "")
        
        # 添加到比较结果
        result["字段"].append(field)
        result["JSON内容"].append(json_value[:80] + "..." if len(str(json_value)) > 80 else json_value)
        result["原始数据"].append(source_value[:80] + "..." if len(str(source_value)) > 80 else source_value)
        result["是否一致"].append("✓" if str(json_value) == str(source_value) else "✗")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="验证JSON captions与原始数据一致性")
    parser.add_argument("--tar_path", type=str, required=True, help="要检查的tar文件路径")
    parser.add_argument("--parquet_path", type=str, help="原始Parquet文件路径")
    parser.add_argument("--csv_path", type=str, help="原始CSV文件路径")
    parser.add_argument("--samples", type=int, default=3, help="要检查的样本数量")
    
    args = parser.parse_args()
    
    if not args.parquet_path and not args.csv_path:
        print("错误：必须提供parquet_path或csv_path中的至少一个")
        return
    
    # 提取随机JSON
    json_samples = extract_random_jsons_from_tar(args.tar_path, args.samples)
    
    for sample in json_samples:
        url = sample['url']
        print(f"\n检查文件: {sample['file']}")
        print(f"URL: {url}")
        
        # 从Parquet或CSV中获取原始数据
        source_data = None
        if args.parquet_path:
            source_data = find_in_parquet(args.parquet_path, url)
            if source_data:
                print(f"在Parquet文件中找到匹配行")
        
        if not source_data and args.csv_path:
            source_data = find_in_csv(args.csv_path, url)
            if source_data:
                print(f"在CSV文件中找到匹配行")
        
        if not source_data:
            print(f"警告：在原始数据源中未找到URL: {url}")
            continue
        
        # 比较caption内容
        comparison = compare_captions(sample['json_data'], source_data)
        
        # 打印表格形式的比较结果
        print("\nCaption比较结果:")
        print(tabulate(comparison, headers="keys", tablefmt="grid"))
        
        # 检查是否全部一致
        all_match = all(mark == "✓" for mark in comparison["是否一致"])
        if all_match:
            print("\n✅ 所有caption字段完全匹配")
        else:
            print("\n❌ 有一些caption字段不匹配")
            mismatch_count = comparison["是否一致"].count("✗")
            print(f"  不匹配字段数: {mismatch_count}/{len(comparison['字段'])}")


if __name__ == "__main__":
    main() 