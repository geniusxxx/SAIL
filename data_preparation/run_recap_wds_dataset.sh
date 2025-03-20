#!/bin/bash

params=(
    -m recap_wds_dataset
    --input_dir /mnt/e/Datasets/cc3m_wds/ 
    --output_dir /mnt/e/Datasets/cc3m_recap_wds/train/ 
    --parquet_path cc3m_3long_3short_1raw_captions_url.parquet 
    --num_test_files 4 
    --num_workers 1
)

# 执行训练命令
python "${params[@]}"