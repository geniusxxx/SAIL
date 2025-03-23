#!/bin/bash

params=(
    -m verify_yfcc_captions
    --parquet_path yfcc15m_3long_3short_1raw_captions_url.parquet 
    --tar_path /mnt/e/Datasets/yfcc15m_wds/000000.tar
    --samples 500
)

# 执行训练命令
python "${params[@]}"