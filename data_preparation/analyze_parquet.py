#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
import time
import json

def analyze_parquet_file(parquet_path, sample_rows=5, output_file=None):
    """
    详细分析parquet文件，检查数据问题
    
    Args:
        parquet_path: parquet文件路径
        sample_rows: 显示每种问题的样本行数
        output_file: 分析结果输出文件路径
    """
    print(f"开始分析parquet文件: {parquet_path}")
    start_time = time.time()
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"读取parquet文件失败: {str(e)}")
        return
    
    print(f"文件读取成功，共 {len(df):,} 行数据")
    
    # 预期的字段列表
    expected_fields = [
        'Image Path', 'raw_caption', 'shortIB_captions', 'longIB_captions',
        'shortSV_captions', 'longSV_captions', 'shortLLA_captions', 'longLLA_captions'
    ]
    
    # 基本信息分析
    print("\n===== 基本信息 =====")
    print(f"文件大小: {os.path.getsize(parquet_path) / (1024 * 1024):.2f} MB")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {', '.join(df.columns.tolist())}")
    
    # 检查列是否存在
    missing_columns = set(expected_fields) - set(df.columns)
    if missing_columns:
        print(f"\n警告: 缺少以下列: {', '.join(missing_columns)}")
    
    # 数据类型分析
    print("\n===== 数据类型分析 =====")
    dtypes = df.dtypes
    for col in df.columns:
        print(f"{col}: {dtypes[col]}")
    
    # 空值分析
    print("\n===== 空值分析 =====")
    null_counts = df.isnull().sum()
    for col in df.columns:
        null_percent = null_counts[col] / len(df) * 100
        if null_counts[col] > 0:
            print(f"{col}: {null_counts[col]:,} 空值 ({null_percent:.2f}%)")
    
    # 数据长度分析
    print("\n===== 字段长度分析 =====")
    length_anomalies = {}
    for col in df.columns:
        if df[col].dtype == 'object':  # 字符串类型
            try:
                # 计算字符串长度
                lengths = df[col].str.len()
                min_len = lengths.min()
                max_len = lengths.max()
                mean_len = lengths.mean()
                
                # 记录极端长度的行
                too_short = df[lengths < mean_len * 0.1].index.tolist()[:sample_rows]
                too_long = df[lengths > mean_len * 10].index.tolist()[:sample_rows]
                
                print(f"{col}: 最短 {min_len}, 最长 {max_len}, 平均 {mean_len:.2f}")
                
                if too_short:
                    print(f"  - 异常短的行索引: {too_short}")
                if too_long:
                    print(f"  - 异常长的行索引: {too_long}")
                
                length_anomalies[col] = {
                    'too_short': too_short,
                    'too_long': too_long
                }
            except Exception as e:
                print(f"{col}: 无法计算长度 - {str(e)}")
    
    # 检查每行是否可以通过字典方式访问
    print("\n===== 行访问测试 =====")
    problematic_rows = []
    
    # 测试iloc和loc访问方式
    row_test_successful = True
    try:
        # 测试第一行
        first_row = df.iloc[0]
        for col in expected_fields:
            if col in df.columns:
                value = first_row[col]
        print("iloc行访问测试通过")
    except Exception as e:
        row_test_successful = False
        print(f"iloc行访问测试失败: {str(e)}")
    
    # 检查所有行数据
    if row_test_successful:
        print("开始检查所有行...")
        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            try:
                # 尝试访问所有期望的字段
                for col in expected_fields:
                    if col in df.columns:
                        _ = row[col]
            except Exception as e:
                problematic_rows.append({
                    'index': idx,
                    'position': i,
                    'error': str(e)
                })
                if len(problematic_rows) <= sample_rows:
                    print(f"行 {i} (索引 {idx}) 访问失败: {str(e)}")
                if len(problematic_rows) == 1:
                    # 对第一个问题行进行详细分析
                    print("\n第一个问题行的详细信息:")
                    try:
                        print(f"问题行数据类型: {type(row)}")
                        print(f"问题行索引类型: {type(idx)}")
                        print(f"问题行内容: {row}")
                    except Exception as ex:
                        print(f"无法打印问题行信息: {str(ex)}")
    
    # 生成摘要报告
    print("\n===== 分析摘要 =====")
    
    has_issues = missing_columns or any(null_counts > 0) or problematic_rows
    if has_issues:
        print("发现数据问题:")
        if missing_columns:
            print(f"- 缺少 {len(missing_columns)} 个必要列")
        
        null_issues = sum(1 for c in null_counts if null_counts[c] > 0)
        if null_issues:
            print(f"- {null_issues} 个列有空值")
        
        if problematic_rows:
            print(f"- 发现 {len(problematic_rows)} 行存在访问问题")
            
        if problematic_rows:
            # 找出问题集中的区域
            if len(problematic_rows) > 1:
                positions = [r['position'] for r in problematic_rows]
                min_pos, max_pos = min(positions), max(positions)
                print(f"- 问题行集中在索引位置 {min_pos} 到 {max_pos}")
                if max_pos - min_pos < len(positions) * 2:
                    print("  (问题行相对集中)")
                else:
                    print("  (问题行分散分布)")
    else:
        print("未发现明显数据问题，parquet文件格式正常。")
    
    # 建议的后续步骤
    if has_issues:
        print("\n===== 建议的解决方案 =====")
        print("1. 可以创建过滤后的parquet文件，移除有问题的行")
        if missing_columns:
            print("2. 处理代码需要适应缺少的列")
        if problematic_rows:
            print(f"3. 有 {len(problematic_rows)} 行需要特别处理或忽略")
    
    # 生成详细报告
    if output_file:
        report = {
            'file_info': {
                'path': parquet_path,
                'size_mb': os.path.getsize(parquet_path) / (1024 * 1024),
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': df.columns.tolist()
            },
            'missing_columns': list(missing_columns),
            'null_counts': {col: int(null_counts[col]) for col in df.columns},
            'length_anomalies': length_anomalies,
            'problematic_rows': problematic_rows[:100],  # 限制数量
            'analysis_time': time.time() - start_time
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存到: {output_file}")
    
    print(f"\n分析完成，耗时 {time.time() - start_time:.2f} 秒")
    return

def filter_parquet_file(input_path, output_path, drop_null=True, fix_problematic=True):
    """
    过滤并修复parquet文件中的问题
    
    Args:
        input_path: 输入parquet文件路径
        output_path: 输出parquet文件路径
        drop_null: 是否删除有空值的行
        fix_problematic: 是否尝试修复问题行
    """
    print(f"开始处理parquet文件: {input_path}")
    start_time = time.time()
    
    # 读取parquet文件
    df = pd.read_parquet(input_path)
    print(f"原始数据: {len(df):,} 行")
    
    # 需要检查的字段列表
    expected_fields = [
        'Image Path', 'raw_caption', 'shortIB_captions', 'longIB_captions',
        'shortSV_captions', 'longSV_captions', 'shortLLA_captions', 'longLLA_captions'
    ]
    
    # 仅保留存在的列
    available_fields = [f for f in expected_fields if f in df.columns]
    
    # 删除空值行
    if drop_null:
        before_count = len(df)
        df = df.dropna(subset=available_fields)
        dropped_count = before_count - len(df)
        print(f"移除了 {dropped_count:,} 行空值数据 ({dropped_count/before_count*100:.2f}%)")
    
    # 修复数据类型问题
    if fix_problematic:
        # 将所有字段转换为字符串
        for field in available_fields:
            # 先填充空值
            df[field] = df[field].fillna('')
            # 转换为字符串
            df[field] = df[field].astype(str)
        print("已将所有字段转换为字符串类型")
    
    # 保存过滤后的数据
    df.to_parquet(output_path)
    print(f"处理后数据: {len(df):,} 行")
    print(f"过滤后的数据已保存至: {output_path}")
    print(f"处理完成，耗时 {time.time() - start_time:.2f} 秒")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="分析和修复parquet文件")
    parser.add_argument("--input", type=str, required=True, help="输入parquet文件路径")
    parser.add_argument("--analyze", action="store_true", help="仅分析文件，不进行修复")
    parser.add_argument("--output", type=str, help="输出parquet文件路径或分析报告路径")
    parser.add_argument("--samples", type=int, default=5, help="每种问题显示的样本数量")
    parser.add_argument("--drop_null", action="store_true", help="删除空值行")
    parser.add_argument("--fix_types", action="store_true", help="修复数据类型问题")
    
    args = parser.parse_args()
    
    if args.analyze:
        output_report = args.output or args.input.replace('.parquet', '_analysis.json')
        analyze_parquet_file(args.input, args.samples, output_report)
    else:
        output_file = args.output or args.input.replace('.parquet', '_fixed.parquet')
        filter_parquet_file(args.input, output_file, args.drop_null, args.fix_types)

if __name__ == "__main__":
    main() 