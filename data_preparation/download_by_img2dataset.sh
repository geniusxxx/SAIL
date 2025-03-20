#!/bin/bash
export NO_ALBUMENTATIONS_UPDATE=1
wandb login --relogin
img2dataset --url_list yfcc15m_3long_3short_1raw_captions_url.parquet \
            --input_format "parquet" \
            --url_col "Image Path" \
            --output_format webdataset \
            --output_folder /mnt/shared_8/dataSet/DreamLIP23M/yfcc15m_recap_wds/train \
            --processes_count 16 \
            --thread_count 128 \
            --resize_mode no \
            --enable_wandb True \
            --tmp_dir /mnt/shared_8/dataSet/DreamLIP23M/yfcc15m_recap_wds/train/tmp \
            --save_additional_columns "[raw_caption,shortIB_captions,longIB_captions,shortSV_captions,longSV_captions,shortLLA_captions,longLLA_captions]"

# img2dataset --url_list cc12m_3long_3short_1raw_captions_url.csv \
#             --input_format "csv" \
#             --url_col "Image Path" \
#             --metadata_col "raw_caption,shortIB_captions,longIB_captions,shortSV_captions,longSV_captions,shortLLA_captions,longLLA_captions" \
#             --output_format webdataset \
#             --output_folder /mnt/shared_8/dataSet/DreamLIP23M/CC12M \
#             --processes_count 20 \
#             --thread_count 32 \
#             --resize_mode no \
#             --enable_wandb True \
#             --tmp_dir /mnt/shared_8/dataSet/DreamLIP23M/CC12M/tmp

# img2dataset --url_list yfcc15m_3long_3short_1raw_captions_url.csv \
#             --input_format "csv" \
#             --url_col "Image Path" \
#             --metadata_col "raw_caption,shortIB_captions,longIB_captions,shortSV_captions,longSV_captions,shortLLA_captions,longLLA_captions" \
#             --output_format webdataset \
#             --output_folder /mnt/shared_8/dataSet/DreamLIP23M/YFCC15M \
#             --processes_count 20 \
#             --thread_count 32 \
#             --resize_mode no \
#             --enable_wandb True \
#             --tmp_dir /mnt/shared_8/dataSet/DreamLIP23M/YFCC15M/tmp