import os
import tarfile
import json
import pandas as pd
import shutil
from tqdm import tqdm
import multiprocessing
import time
import hashlib
import argparse
from pathlib import Path
import tempfile
import stat
import subprocess

def verify_paths_safety(input_dir, output_dir):
    """验证输入输出路径安全性"""
    # 确保路径不同
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError("输入和输出路径不能相同，以保护原始数据")
    
    # 验证输入路径存在且可读
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    if not os.access(input_dir, os.R_OK):
        raise PermissionError(f"无法读取输入目录: {input_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    return True

def get_file_hash(file_path):
    """获取文件的MD5哈希，用于临时目录命名，避免冲突"""
    hash_obj = hashlib.md5()
    hash_obj.update(file_path.encode('utf-8'))
    hash_obj.update(str(time.time()).encode('utf-8'))
    return hash_obj.hexdigest()[:10]

def create_url_to_captions_map(parquet_path):
    """从parquet文件创建URL到captions的映射，具有错误处理能力"""
    print(f"正在加载parquet文件: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"读取完成，共 {len(df)} 条记录")
    
    url_to_captions = {}
    skipped_rows = 0
    error_positions = []
    
    # 使用更健壮的to_dict('records')方法进行迭代
    for i, record in enumerate(tqdm(df.to_dict('records'), total=len(df), desc="创建URL映射")):
        try:
            url = record['Image Path']
            # 保存为单元素数组格式，符合WebDataset加载要求
            url_to_captions[url] = {
                'raw_caption': [record['raw_caption']],
                'shortIB_captions': [record['shortIB_captions']],
                'longIB_captions': [record['longIB_captions']],
                'shortSV_captions': [record['shortSV_captions']],
                'longSV_captions': [record['longSV_captions']],
                'shortLLA_captions': [record['shortLLA_captions']],
                'longLLA_captions': [record['longLLA_captions']]
            }
        except Exception as e:
            skipped_rows += 1
            # 只记录前10个错误位置，避免输出过多
            if len(error_positions) < 10:
                error_positions.append(i)
                print(f"跳过第 {i} 行记录，错误: {str(e)}")
            # 每1000个错误只显示一次汇总
            elif skipped_rows % 1000 == 0:
                print(f"已跳过 {skipped_rows} 个问题行...")
            continue
    
    # 显示跳过统计
    if skipped_rows > 0:
        print(f"\n处理过程中共跳过 {skipped_rows} 行有问题的记录")
        if error_positions:
            print(f"前几个问题行的位置: {error_positions}")
    
    print(f"URL映射创建完成，共 {len(url_to_captions)} 个有效URL")
    return url_to_captions

def process_single_tar(original_tar_path, new_tar_path, url_to_captions_map, stats, unmatched_urls=None):
    """处理单个tar文件，创建新的tar文件"""
    # 创建唯一的临时目录名称 - 使用系统临时目录
    tar_basename = os.path.basename(original_tar_path)
    temp_dir = tempfile.mkdtemp(prefix=f"recap_wds_{get_file_hash(original_tar_path)}_", dir="/tmp")
    print(f"创建临时目录: {temp_dir}")
    
    # 如果提供了unmatched_urls，则收集未匹配的URL
    local_unmatched_urls = []
    
    try:
        # 1. 提取原tar文件
        print(f"正在提取: {original_tar_path}")
        with tarfile.open(original_tar_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        # 设置临时目录的权限为完全读写执行
        print(f"修改目录权限: {temp_dir}")
        os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 777权限
        
        # 使用bash命令修改所有文件的权限
        try:
            subprocess.run(f"chmod -R 666 {temp_dir}/*", shell=True, check=True)
            print("已修改所有文件权限")
        except subprocess.SubprocessError as e:
            print(f"修改文件权限时出错: {e}")
        
        # 2. 找出所有文件组
        file_groups = {}  # 按前缀分组
        for filename in os.listdir(temp_dir):
            prefix = filename.split('.')[0]
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append(filename)
        
        # 3. 处理每个文件组
        processed_count = 0
        matched_count = 0
        total_groups = len(file_groups)
        
        for prefix, files in tqdm(file_groups.items(), total=total_groups, desc=f"处理 {tar_basename}"):
            # 找到该组的json和图片文件
            json_file = next((f for f in files if f.endswith('.json')), None)
            image_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            
            if json_file and image_files:
                processed_count += 1
                # 读取并修改JSON
                json_path = os.path.join(temp_dir, json_file)
                try:
                    # 确保JSON文件有读写权限
                    os.chmod(json_path, 0o666)  # 设置为666权限

                    with open(json_path, 'r', encoding='utf-8') as f:
                        try:
                            json_data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"警告: JSON解析错误 {json_path}")
                            continue
                    
                    # 获取URL并在映射中查找captions
                    url = json_data.get('url', '')
                    if url in url_to_captions_map:
                        matched_count += 1
                        captions_data = url_to_captions_map[url]
                        
                        # 删除原caption字段
                        if 'caption' in json_data:
                            del json_data['caption']
                        
                        # 添加7类caption数组
                        json_data.update(captions_data)
                        
                        # 写回JSON文件
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, ensure_ascii=False)
                    else:
                        # 记录未匹配的URL
                        if unmatched_urls is not None:
                            uid = prefix  # 使用文件前缀作为UID
                            local_unmatched_urls.append({
                                'uid': uid,
                                'url': url,
                                'tar_file': tar_basename
                            })
                        
                except Exception as e:
                    print(f"处理JSON文件 {json_path} 时出错: {str(e)}")
                    continue
        
        # 4. 创建新tar文件
        print(f"创建新tar文件: {new_tar_path}")
        # 确保输出目录存在
        os.makedirs(os.path.dirname(new_tar_path), exist_ok=True)
        
        with tarfile.open(new_tar_path, 'w') as new_tar:
            for prefix, files in file_groups.items():
                # 只添加图片和修改后的JSON文件
                for filename in files:
                    if filename.endswith(('.jpg', '.png', '.jpeg', '.webp', '.json')):
                        file_path = os.path.join(temp_dir, filename)
                        if os.path.exists(file_path):  # 确保文件存在
                            try:
                                new_tar.add(file_path, arcname=filename)
                            except Exception as e:
                                print(f"添加文件到tar时出错 {file_path}: {str(e)}")
        
        # 更新统计信息
        stats['total_processed'] += processed_count
        stats['total_matched'] += matched_count
        stats['match_rate'] = stats['total_matched'] / stats['total_processed'] if stats['total_processed'] > 0 else 0
        
        # 返回本地收集的未匹配URL
        if unmatched_urls is not None:
            unmatched_urls.extend(local_unmatched_urls)
        
        print(f"处理完成: {tar_basename}, 处理: {processed_count}, 匹配: {matched_count}, 未匹配: {processed_count - matched_count}")
        return processed_count, matched_count
    
    except Exception as e:
        print(f"错误处理 {original_tar_path}: {str(e)}")
        return 0, 0
    
    finally:
        # 5. 清理临时目录
        print(f"清理临时目录: {temp_dir}")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"清理临时目录时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="将cc12m_wds转换为cc12m_recap_wds格式")
    parser.add_argument("--input_dir", type=str, default="/mnt/e/Datesets/cc12m_wds", 
                        help="原始wds数据集目录")
    parser.add_argument("--output_dir", type=str, default="/mnt/e/Datesets/cc12m_recap_wds/train", 
                        help="新数据集输出目录")
    parser.add_argument("--parquet_path", type=str, required=True,
                        help="包含7类caption的Parquet文件路径")
    parser.add_argument("--num_test_files", type=int, default=4,
                        help="要处理的测试文件数量")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="并行处理的worker数量")
    parser.add_argument("--tmp_dir", type=str, default="/tmp",
                        help="临时文件目录（默认为/tmp，应该是Linux文件系统）")
    parser.add_argument("--unmatched_output", type=str, default="unmatched_urls.csv",
                        help="未匹配URL的输出文件路径")
    
    args = parser.parse_args()
    
    # 验证临时目录是否在Linux文件系统上
    if args.tmp_dir.startswith('/mnt/'):
        print(f"警告: 临时目录 {args.tmp_dir} 似乎在Windows挂载点上，这可能导致性能和权限问题")
        print("建议使用Linux原生文件系统目录，如 /tmp")
        choice = input("是否继续? [y/N]: ")
        if choice.lower() != 'y':
            print("已取消")
            return
    
    # 验证路径安全性
    verify_paths_safety(args.input_dir, args.output_dir)
    
    # 获取所有tar文件
    tar_files = [f for f in os.listdir(args.input_dir) if f.endswith('.tar')]
    tar_files.sort()  # 排序以确保结果一致
    
    # 选择前N个作为测试
    selected_tars = tar_files[:args.num_test_files]
    print(f"选择的tar文件: {selected_tars}")
    
    # 创建URL到captions的映射
    url_to_captions_map = create_url_to_captions_map(args.parquet_path)
    
    # 共享统计信息
    manager = multiprocessing.Manager()
    stats = manager.dict()
    stats['total_processed'] = 0
    stats['total_matched'] = 0
    stats['match_rate'] = 0
    
    # 收集未匹配的URL列表
    unmatched_urls = []
    
    # 创建任务列表
    tasks = []
    for tar_file in selected_tars:
        original_tar_path = os.path.join(args.input_dir, tar_file)
        new_tar_path = os.path.join(args.output_dir, tar_file)
        tasks.append((original_tar_path, new_tar_path, url_to_captions_map, stats, unmatched_urls))
    
    # 串行或并行处理
    if args.num_workers <= 1:
        # 串行处理
        for task in tasks:
            process_single_tar(*task)
    else:
        # 并行处理时需要使用Manager列表
        unmatched_urls = manager.list()
        tasks = [(t[0], t[1], t[2], t[3], unmatched_urls) for t in tasks]
        with multiprocessing.Pool(args.num_workers) as pool:
            pool.starmap(process_single_tar, tasks)
        # 转换为普通列表
        unmatched_urls = list(unmatched_urls)
    
    # 打印最终统计
    print("\n转换完成!\n")
    print(f"总处理样本数: {stats['total_processed']}")
    print(f"成功匹配样本数: {stats['total_matched']}")
    print(f"匹配率: {stats['match_rate']*100:.2f}%")
    
    # 将未匹配的URL写入CSV文件
    if unmatched_urls:
        unmatched_file = args.unmatched_output
        print(f"\n保存未匹配URL到文件: {unmatched_file}")
        unmatched_df = pd.DataFrame(unmatched_urls)
        unmatched_df.to_csv(unmatched_file, index=False)
        print(f"共保存 {len(unmatched_urls)} 个未匹配的URL")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"总耗时: {elapsed_time:.2f}秒")