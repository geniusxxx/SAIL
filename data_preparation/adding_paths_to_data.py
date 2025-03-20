import argparse
import csv
import os
from tqdm import tqdm

def add_image_paths(input_csv_file, chunk_size=50000):
    images_per_folder = 10000
    
    print('Reading CSV file:', input_csv_file)
    temp_output = input_csv_file + '.temp'
    
    try:
        # 先读取一遍实际的数据行数
        print("Counting actual data rows...")
        with open(input_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过header
            actual_lines = sum(1 for _ in reader)
        print(f"Actual data rows: {actual_lines}")
        
        dataset_name = input_csv_file.split('_')[0]
        processed = 0
        
        with open(input_csv_file, 'r', encoding='utf-8') as infile, \
             open(temp_output, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # 处理header
            header = next(reader)
            print(f"Header: {header}")
            if 'Image Path' in header:
                header[header.index('Image Path')] = 'Image Url'
            header.append('Image Path')
            writer.writerow(header)
            
            # 使用缓冲区处理数据
            buffer = []
            pbar = tqdm(total=actual_lines, desc="Processing")
            
            for row in reader:
                # 生成图片路径
                folder = processed // images_per_folder
                index = processed % images_per_folder
                img_path = f"{dataset_name}/images/{folder:07d}/{index:07d}.jpg"
                
                # 添加路径
                row.append(img_path)
                buffer.append(row)
                processed += 1
                pbar.update(1)
                
                # 当缓冲区达到指定大小时写入文件
                if len(buffer) >= chunk_size:
                    writer.writerows(buffer)
                    buffer = []
            
            # 写入剩余的行
            if buffer:
                writer.writerows(buffer)
            
            pbar.close()
        
        # 验证结果
        result_rows = sum(1 for _ in open(temp_output)) - 1
        print(f"\nProcessed rows: {processed}")
        print(f"Result rows: {result_rows}")
        print(f"Expected rows: {actual_lines}")
        
        if processed == actual_lines and result_rows == actual_lines:
            os.replace(temp_output, input_csv_file)
            print('Successfully updated file!')
        else:
            os.remove(temp_output)
            raise Exception(f"Row count mismatch: processed {processed}, expected {actual_lines}")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add image paths to CSV file')
    parser.add_argument('--input_csv_file', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--chunk_size', type=int, default=50000, help='Number of rows to process at once')
    
    args = parser.parse_args()
    add_image_paths(args.input_csv_file, args.chunk_size)