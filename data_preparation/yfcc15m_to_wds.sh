#!/bin/bash

params=(
    -m yfcc15m_to_wds
    --input /mnt/e/Datasets/yfcc15m_parquet/train-00001-of-01507.parquet 
    --output /mnt/e/Datasets/yfcc15m_wds
    --samples_per_tar 10000
)

# 执行训练命令
python "${params[@]}"