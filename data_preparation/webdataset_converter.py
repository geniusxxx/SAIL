import os
import io
import json
import tarfile
import pandas as pd
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

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