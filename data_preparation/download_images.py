import argparse
import pandas as pd
import numpy as np
import requests
import zlib
import os
import shelve
import magic
import time
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import io
import json
import hashlib

# 导入WebDataset转换器
# 如果需要WebDataset转换功能，使用以下导入
# from webdataset_converter import convert_to_webdataset

class DownloadStats:
    """下载统计类"""
    def __init__(self, save_dir):
        self.failed_count = 0
        self.success_count = 0
        self.failed_urls = []
        self.total_bytes = 0
        self.start_time = time.time()
        self.save_dir = save_dir
        self.failed_log_path = os.path.join(save_dir, 'failed_downloads.csv')
        self.detailed_log_path = os.path.join(save_dir, 'download_details.csv')
        
        # 创建或清空失败日志文件
        with open(self.failed_log_path, 'w', encoding='utf-8') as f:
            f.write("Time,URL,Status,Error\n")
            
        # 创建详细日志文件
        with open(self.detailed_log_path, 'w', encoding='utf-8') as f:
            f.write("Time,URL,Status,Path,Size,MimeType\n")
            
    def update(self, row):
        """更新统计信息"""
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        
        if row['status'] != 200:
            self.failed_count += 1
            error_msg = row.get('error_message', '')
            
            # 添加到内存中的失败列表
            self.failed_urls.append({
                'url': row['url'],
                'status': row['status'],
                'time': current_time,
                'error': error_msg
            })
            
            # 实时写入失败记录到文件
            with open(self.failed_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{current_time},{row['url']},{row['status']},{error_msg}\n")
        else:
            self.success_count += 1
            # 累计下载字节数（如果有）
            if 'size' in row:
                self.total_bytes += row['size']
            
        # 记录所有下载详情（无论成功或失败）
        with open(self.detailed_log_path, 'a', encoding='utf-8') as f:
            path = row.get('file', '')
            size = row.get('size', 0)
            mime = row.get('mimetype', '')
            f.write(f"{current_time},{row['url']},{row['status']},{path},{size},{mime}\n")
            
    def print_stats(self):
        """打印当前统计信息"""
        total = self.success_count + self.failed_count
        elapsed_time = time.time() - self.start_time
        
        if total > 0:
            success_rate = (self.success_count / total) * 100
            avg_speed = self.total_bytes / (elapsed_time + 0.001) / 1024 / 1024  # MB/s
            
            print(f"\n当前下载统计:")
            print(f"总数: {total}")
            print(f"成功: {self.success_count} ({success_rate:.2f}%)")
            print(f"失败: {self.failed_count} ({100-success_rate:.2f}%)")
            print(f"平均速度: {avg_speed:.2f} MB/s")
            print(f"已用时间: {elapsed_time:.1f}秒")
            print(f"失败记录保存在: {self.failed_log_path}")
            print(f"详细记录保存在: {self.detailed_log_path}")

# # Parse command line arguments
# headers = {
#     'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
#     'X-Forwarded-For': '64.18.15.200'
# }
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)

def df_multiprocess(df, processes, chunk_size, func, dataset_name, save_dir='.'):
    """
    多进程处理DataFrame并确保数据正确写入
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 构造临时文件路径
    tmp_file_path = os.path.join(save_dir, '%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size))
    print(f"临时文件将保存到: {tmp_file_path}")
    
    print("生成数据分片...")
    
    # 初始化统计对象
    stats = DownloadStats(save_dir)
    
    with shelve.open(tmp_file_path, writeback=True) as results:
        # 创建更丰富的进度条
        pbar = tqdm(
            total=len(df), 
            position=0,
            desc="准备中",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            postfix={"成功": 0, "失败": 0, "成功率": "0%"}
        )
        
        # 断点续传
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "继续下载"
        for k in results.keys():
            chunk_data = results[str(k)][1]
            pbar.update(len(chunk_data))
            
            # 更新统计信息
            for _, row in chunk_data.iterrows():
                stats.update(row)
                
            print(f"已加载已完成的chunk {k}, 包含 {len(chunk_data)} 条记录")

        # 生成待处理的数据分片
        pool_data = [(index, df[i:i + chunk_size], func) 
                    for index, i in enumerate(range(0, len(df), chunk_size)) 
                    if index not in finished_chunks]
        
        print(f"待处理分片数: {len(pool_data)}, 每个分片大小: {chunk_size}")

        pbar.desc = "下载中"
        with Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                try:
                    chunk_index = str(result[0])
                    chunk_data = result[1]
                    
                    # 更新统计信息
                    success_in_chunk = 0
                    failure_in_chunk = 0
                    
                    for _, row in chunk_data.iterrows():
                        stats.update(row)
                        if row['status'] == 200:
                            success_in_chunk += 1
                        else:
                            failure_in_chunk += 1
                    
                    # 更新进度条显示
                    success_count = stats.success_count
                    failed_count = stats.failed_count
                    total_count = success_count + failed_count
                    success_rate = success_count / (total_count or 1) * 100
                    
                    pbar.set_postfix(
                        成功=success_count, 
                        失败=failed_count, 
                        成功率=f"{success_rate:.1f}%"
                    )
                    
                    # 每处理100张图片打印一次统计
                    if total_count % 100 == 0:
                        stats.print_stats()
                    
                    # 写入数据并立即同步
                    results[chunk_index] = result
                    results.sync()
                    
                    print(f"成功写入chunk {chunk_index}, "
                          f"包含 {len(chunk_data)} 条记录 "
                          f"(成功: {success_in_chunk}, "
                          f"失败: {failure_in_chunk})")
                    
                    pbar.update(len(chunk_data))
                except Exception as e:
                    print(f"写入chunk {result[0]} 失败: {str(e)}")
                    continue
        
        pbar.close()
        
        # 打印最终统计信息
        print("\n下载完成，最终统计:")
        stats.print_stats()

    return df_from_shelve(chunk_size, func, dataset_name, save_dir)

# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row['file'])):
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
    return row

# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    try:
        # not all sites will support HEAD
        response = requests.head(row['url'], stream=False, timeout=5, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except Exception as e:
        # 记录具体错误信息
        row['status'] = 408  # 超时错误
        row['error_message'] = str(e)
    return row

def download_image(row):
    fname = os.path.join(row['save_dir'], row['Image Path'])
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # 跳过已下载的文件
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        try:
            row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except Exception as e:
            row['status'] = 500  # 文件可能损坏
            row['error_message'] = f"文件已存在但损坏: {str(e)}"
        return row

    # 添加重试机制
    max_retries = 3
    retry_delay = 1
    
    for retry in range(max_retries):
        try:
            response = requests.get(
                row['url'],
                stream=True,  # 启用流式下载
                timeout=(3, 10),  # 连接超时3秒，读取超时10秒
                allow_redirects=True,
                headers=headers
            )
            row['status'] = response.status_code

            if response.ok:
                try:
                    start_time = time.time()
                    max_download_time = 30  # 最大下载时间30秒
                    total_size = 0
                    
                    with open(fname, 'wb') as out_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if time.time() - start_time > max_download_time:
                                raise TimeoutError("Download took too long")
                            if chunk:
                                total_size += len(chunk)
                                out_file.write(chunk)
                    
                    row['mimetype'] = magic.from_file(fname, mime=True)
                    row['size'] = os.stat(fname).st_size
                    row['file'] = fname
                    row['downloaded_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    return row
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay * (retry + 1))  # 指数退避
                        continue
                    row['status'] = 408
                    row['error_message'] = f"下载中断: {str(e)}"
                    return row
            else:
                row['error_message'] = f"HTTP错误: {response.status_code}"
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(retry_delay * (retry + 1))
                continue
            row['status'] = 408
            row['error_message'] = f"请求异常: {str(e)}"
            return row
    
    return row

def open_csv(fname, folder, save_dir, start_index=None, end_index=None):
    print("打开数据文件 %s..." % fname)
    df = pd.read_csv(fname)
    
    # 确保包含必要列
    if 'url' not in df.columns:
        df['url'] = df['Image Path']
    
    # 使用MD5哈希替代Python的hash函数
    def get_md5_filename(url):
        # 使用MD5算法对URL进行哈希
        md5_hash = hashlib.md5(url.encode()).hexdigest()
        return f"{md5_hash}.jpg"
    
    # 应用新的哈希命名方式
    df['Image Path'] = df['url'].apply(get_md5_filename)
    
    # 添加save_dir列
    df['save_dir'] = save_dir
    
    if start_index is not None and end_index is not None:
        print("切片数据框，从 %d 到 %d" % (start_index, end_index))
        df = df.iloc[start_index:end_index]
    print("处理", len(df), "张图像:")
    return df

def df_from_shelve(chunk_size, func, dataset_name, save_dir='.'):
    """从shelve文件生成DataFrame"""
    tmp_file_path = os.path.join(save_dir, '%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size))
    print(f"从临时文件读取结果: {tmp_file_path}")
    
    print("从结果生成数据框...")
    with shelve.open(tmp_file_path) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='下载图像并转换为WebDataset格式')
    parser.add_argument('--csv_path', type=str, required=True, help='包含图像URL的CSV文件路径')
    parser.add_argument('--start_index', type=int, default=0, help='处理CSV文件的起始索引')
    parser.add_argument('--end_index', type=int, default=None, help='处理CSV文件的结束索引') 
    parser.add_argument('--num_processes', type=int, default=64, help='使用的并行进程数')
    parser.add_argument('--chunk_size', type=int, default=500, help='每个进程每块的图像数')
    parser.add_argument('--data_name', type=str, default='training', help='数据集名称')
    
    # 修改目录参数以符合用户需求
    parser.add_argument('--save_dir', type=str, required=True, help='图像文件保存目录')
    parser.add_argument('--tmp_dir', type=str, default=None, help='临时文件保存目录，默认为save_dir下的tmp子目录')
    parser.add_argument('--wds_dir', type=str, default=None, help='WebDataset输出目录')
    
    parser.add_argument('--convert_to_wds', action='store_true', help='下载后是否转换为WebDataset格式')
    parser.add_argument('--shard_size', type=int, default=1000, help='每个WebDataset分片的样本数')
    parser.add_argument('--wds_processes', type=int, default=8, help='WebDataset转换使用的进程数')
    args = parser.parse_args()

    # 设置默认目录（如果未指定）
    if args.tmp_dir is None:
        args.tmp_dir = os.path.join(args.save_dir, 'tmp')
    
    if args.wds_dir is None and args.convert_to_wds:
        args.wds_dir = args.save_dir + '_wds'
    
    # 确保所有目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    if args.convert_to_wds:
        os.makedirs(args.wds_dir, exist_ok=True)
    
    print(f"图像将保存到: {args.save_dir}")
    print(f"临时文件将保存到: {args.tmp_dir}")
    if args.convert_to_wds:
        print(f"WebDataset将输出到: {args.wds_dir}")

    # 第一步：下载图像
    df = open_csv(args.csv_path, args.data_name, args.save_dir, args.start_index, args.end_index)
    df_multiprocess(
        df=df, 
        processes=args.num_processes, 
        chunk_size=args.chunk_size, 
        func=download_image, 
        dataset_name=args.data_name,
        save_dir=args.tmp_dir  # 传递临时文件目录
    )
    
    df = df_from_shelve(
        chunk_size=args.chunk_size, 
        func=download_image, 
        dataset_name=args.data_name,
        save_dir=args.tmp_dir  # 传递临时文件目录
    )
    
    # 保存下载报告到tmp_dir
    report_path = os.path.join(args.tmp_dir, f"downloaded_{args.data_name}_report.csv")
    df.to_csv(report_path, index=False)
    print(f"下载报告已保存到 {report_path}")

    # 第二步：如果需要，转换为WebDataset格式
    if args.convert_to_wds:
        try:
            from webdataset_converter import convert_to_webdataset
            convert_to_webdataset(
                df=df, 
                output_path=os.path.join(args.wds_dir, args.data_name),
                dataset_name=args.data_name,
                num_processes=args.wds_processes,
                shard_size=args.shard_size
            )
        except ImportError:
            print("错误: 未能导入webdataset_converter模块，请确保webdataset_converter.py文件在同一目录中")
