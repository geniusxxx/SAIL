#!/bin/bash

params=(
    -m analyze_parquet
    --input cc3m_3long_3short_1raw_captions_url.parquet 
    --analyze
)

# 执行训练命令
python "${params[@]}"