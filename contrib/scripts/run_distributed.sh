#!/bin/bash

# number of GPUs to train on
NGPUS=$(nvidia-smi -L | wc -l)
if [ "$NGPUS" -lt "2" ]; then
    echo "ERROR: cannot run distributed training without 2 or more GPUs."
    exit 1
fi
# specify pretrained HRNet backbone
PRETRAINED_HRNET='/home/alfred/models/hrnetv2_w48_imagenet_pretrained.pth'
# DATA_F3='/home/alfred/data/dutch/data'
# DATA_PENOBSCOT='/home/alfred/data/penobscot'
DATA_F3='/storage/data/dutchf3/data'
DATA_PENOBSCOT='/storage/data/penobscot'
# subdirectory where results are written
OUTPUT_DIR='output'

unset CUDA_VISIBLE_DEVICES
# bug to fix conda not launching from a bash shell
source /data/anaconda/etc/profile.d/conda.sh
conda activate seismic-interpretation

cd experiments/interpretation/dutchf3_patch/

# patch based without skip connections
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'none' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'no_depth' \
    --cfg=configs/patch_deconvnet.yaml > patch_deconvnet.log 2>&1

# patch based with skip connections
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'none' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'no_depth' \
    --cfg=configs/patch_deconvnet_skip.yaml > patch_deconvnet_skip.log 2>&1

# squeeze excitation resnet unet + section depth
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'section' \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg=configs/seresnet_unet.yaml > seresnet_unet.log 2>&1

# HRNet + patch depth
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'patch' \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'patch_depth' \
    --cfg=configs/hrnet.yaml > hrnet_patch.log 2>&1

# HRNet + section depth
nohup time python -m torch.distributed.launch --nproc_per_node=${NGPU} train.py \
    'DATASET.ROOT' "${DATA_F3}" \
    'TRAIN.DEPTH' 'section' \
    'MODEL.PRETRAINED' "${PRETRAINED_HRNET}" \
    'OUTPUT_DIR' "${OUTPUT_DIR}" 'TRAIN.MODEL_DIR' 'section_depth' \
    --cfg=configs/hrnet.yaml > hrnet_section.log 2>&1

echo "TADA"
