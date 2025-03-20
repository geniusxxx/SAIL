import argparse
import pandas as pd
import numpy as np
import requests
import zlib
import os
import shelve
import magic
import time  # 添加time模块
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import io
import json
import tarfile

# Parse command line arguments
headers = {
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}

def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)

def df_multiprocess(df, processes, chunk_size, func, dataset_name, save_dir='.'):
    """
    多进程处理DataFrame
    
    参数:
    df - 数据框
    processes - 进程数
    chunk_size - 每块处理的记录数
    func - 处理函数
    dataset_name - 数据集名称
    save_dir - 保存临时文件的目录，默认为当前目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 构造临时文件路径
    tmp_file_path = os.path.join(save_dir, '%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size))
    print(f"临时文件将保存到: {tmp_file_path}")
    
    print("生成数据分片...")
    with shelve.open(tmp_file_path) as results:
        pbar = tqdm(total=len(df), position=0)
        # 断点续传:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "继续下载"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        print(int(len(df) / chunk_size), "个分片。", "每个分片", chunk_size, "条记录。", "使用", processes, "个进程")

        pbar.desc = "下载中"
        with Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("下载完成.")
    return


# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row['file'])):
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
    return row

# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(row['url'], stream=False, timeout=5, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
    if response.ok:
        row['file'] = fname
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
        except:
            row['status'] = 500  # 文件可能损坏
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
                    
                    with open(fname, 'wb') as out_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if time.time() - start_time > max_download_time:
                                raise TimeoutError("Download took too long")
                            if chunk:
                                out_file.write(chunk)
                    
                    row['mimetype'] = magic.from_file(fname, mime=True)
                    row['size'] = os.stat(fname).st_size
                    row['file'] = fname
                    return row
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay * (retry + 1))  # 指数退避
                        continue
                    row['status'] = 408
                    return row
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(retry_delay * (retry + 1))
                continue
            row['status'] = 408
            return row
    
    return row

def create_sample_for_webdataset(row):
    """为每个图像创建一个WebDataset样本，保留所有文本描述"""
    try:
        image_path = row['file']
        
        # 读取图像
        img = Image.open(image_path)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=img.format or 'JPEG')
        img_bytes = img_bytes.getvalue()
        
        # 创建包含所有可能文本描述的元数据
        metadata = {
            'url': row['url']
        }
        
        # 添加所有标准文本字段，如果存在的话
        text_fields = [
            'raw_caption',
            'shortIB_captions', 'longIB_captions',
            'shortSV_captions', 'longSV_captions',
            'shortLLA_captions', 'longLLA_captions'
        ]
        
        for field in text_fields:
            if field in row:
                metadata[field] = row[field]
        
        # 兼容处理其他可能的字段命名
        alt_fields = {
            'caption': 'raw_caption',
            'long_caption_0': 'longIB_captions',
            'long_caption_1': 'longSV_captions',
            'long_caption_2': 'longLLA_captions',
            'short_caption_0': 'shortIB_captions',
            'short_caption_1': 'shortSV_captions',
            'short_caption_2': 'shortLLA_captions'
        }
        
        for alt_field, std_field in alt_fields.items():
            if alt_field in row and std_field not in metadata:
                metadata[std_field] = row[alt_field]
        
        # 添加其他有用的元数据，如图像尺寸、状态等
        additional_fields = [
            'key', 'status', 'error_message', 
            'width', 'height', 'original_width', 'original_height',
            'sha256', 'exif'
        ]
        
        for field in additional_fields:
            if field in row:
                metadata[field] = row[field]
        
        # 如果没有key字段，使用文件名作为键
        if 'key' not in metadata:
            metadata['key'] = os.path.basename(image_path).split('.')[0]
        
        # 创建JSON
        json_data = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
        
        # 创建样本ID（优先使用key字段，其次使用图像名的哈希值）
        sample_id = metadata.get('key', os.path.basename(image_path).split('.')[0])
        
        return {
            'success': True,
            'id': sample_id,
            'img': img_bytes,
            'json': json_data
        }
    except Exception as e:
        # 出错时返回错误信息
        error_id = 'unknown'
        if 'image_path' in locals():
            error_id = os.path.basename(image_path).split('.')[0]
        elif 'row' in locals() and 'key' in row:
            error_id = row['key']
            
        return {
            'success': False,
            'id': error_id,
            'error': str(e)
        }

def open_csv(fname, folder, save_dir, start_index=None, end_index=None):
    print("打开数据文件 %s..." % fname)
    df = pd.read_csv(fname)
    
    # 确保包含必要列
    if 'Image Url' in df.columns:
        df = df.rename(columns={'Image Url': 'url'})
    elif 'url' not in df.columns:
        # 尝试其他可能的列名
        for col in df.columns:
            if 'url' in col.lower():
                df = df.rename(columns={col: 'url'})
                break
    
    # 图像路径相对于save_dir，而不是绝对路径
    if 'Image Path' not in df.columns:
        if 'key' in df.columns:
            # 使用key作为文件名但不包含save_dir前缀
            df['Image Path'] = df.apply(lambda row: os.path.join(folder, f"{row['key']}.jpg"), axis=1)
        else:
            # 使用URL哈希作为文件名
            df['Image Path'] = df.apply(lambda row: os.path.join(folder, f"{hash(row['url'])}.jpg"), axis=1)
    
    # 添加save_dir列，供download_image使用
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

def convert_to_webdataset(df, output_path, dataset_name, num_processes=8, shard_size=1000):
    """将下载的图像转换为WebDataset格式
    
    参数:
    df - 包含图像信息的DataFrame (必须包含'file'列和图像路径)
    output_path - WebDataset输出目录
    dataset_name - 数据集名称，用于创建分片文件名
    num_processes - 用于转换的进程数
    shard_size - 每个WebDataset分片中的样本数量
    """
    print(f"转换为WebDataset格式，输出到 {output_path}")
    
    # 筛选有效图像
    valid_df = filter_valid_images(df)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 准备多进程转换
    batch_size = shard_size * 10  # 每个进程处理10个分片
    batches = []
    for i in range(0, len(valid_df), batch_size):
        batches.append(valid_df.iloc[i:i+batch_size])
    
    print(f"将数据分成 {len(batches)} 个批次进行处理")
    
    all_shards = []
    with Pool(num_processes) as pool:
        batch_args = [(batch, output_path, f"{dataset_name}", shard_size) for batch in batches]
        for i, result in enumerate(tqdm(pool.starmap(process_batch_to_webdataset, batch_args), total=len(batches), desc="转换为WebDataset")):
            all_shards.extend(result)
    
    # 创建索引文件
    with open(os.path.join(output_path, f"{dataset_name}_shards.txt"), "w") as f:
        for shard in all_shards:
            f.write(f"{os.path.basename(shard)}\n")
    
    print(f"转换完成，共创建 {len(all_shards)} 个WebDataset分片")
    return all_shards

def filter_valid_images(df):
    """筛选有效的图像（成功下载且是图像类型）"""
    print(f"开始筛选有效图像，初始数量: {len(df)}")
    
    # 只保留状态为200的行
    filtered_df = df[df['status'] == 200].copy()
    print(f"HTTP状态码为200的图像: {len(filtered_df)}")
    
    # 确保file列存在
    if 'file' not in filtered_df.columns:
        # 如果没有file列但有Image Path列，使用它
        if 'Image Path' in filtered_df.columns:
            if 'save_dir' in filtered_df.columns:
                # 如果有save_dir列，构建完整路径
                filtered_df['file'] = filtered_df.apply(
                    lambda row: os.path.join(row['save_dir'], row['Image Path']), axis=1
                )
            else:
                # 否则直接使用Image Path
                filtered_df['file'] = filtered_df['Image Path']
    
    # 检查文件是否存在
    filtered_df = filtered_df[filtered_df['file'].apply(lambda x: os.path.isfile(x))]
    print(f"文件存在的图像: {len(filtered_df)}")
    
    # 检查mimetype以确保是图像（如果mimetype列存在）
    if 'mimetype' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['mimetype'].str.startswith('image/')]
        print(f"MIME类型为图像的文件: {len(filtered_df)}")
    
    # 检查文件大小（如果size列存在）
    if 'size' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['size'] > 1024]  # 大于1KB
        print(f"大小合适的图像: {len(filtered_df)}")
    
    print(f"筛选后剩余 {len(filtered_df)} 张有效图像，占原始数据的 {len(filtered_df)/len(df)*100:.2f}%")
    return filtered_df

def process_batch_to_webdataset(batch_df, output_path, shard_prefix, shard_size=1000):
    """将一批数据处理成WebDataset分片"""
    os.makedirs(output_path, exist_ok=True)
    
    # 将数据分成更小的分片
    shards = []
    for i in range(0, len(batch_df), shard_size):
        shard_df = batch_df.iloc[i:i+shard_size]
        shards.append(shard_df)
    
    results = []
    for shard_idx, shard_df in enumerate(shards):
        shard_path = os.path.join(output_path, f"{shard_prefix}_{shard_idx:06d}.tar")
        
        # 创建tar文件
        with tarfile.open(shard_path, "w") as tar:
            for _, row in shard_df.iterrows():
                sample = create_sample_for_webdataset(row)
                if not sample['success']:
                    continue
                
                sample_id = sample['id']
                
                # 添加图像到tar
                img_info = tarfile.TarInfo(f"{sample_id}.jpg")
                img_info.size = len(sample['img'])
                tar.addfile(img_info, io.BytesIO(sample['img']))
                
                # 添加JSON到tar
                json_info = tarfile.TarInfo(f"{sample_id}.json")
                json_info.size = len(sample['json'])
                tar.addfile(json_info, io.BytesIO(sample['json']))
        
        results.append(shard_path)
    
    return results

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
        convert_to_webdataset(
            df=df, 
            output_path=os.path.join(args.wds_dir, args.data_name),
            dataset_name=args.data_name,
            num_processes=args.wds_processes,
            shard_size=args.shard_size
        )
