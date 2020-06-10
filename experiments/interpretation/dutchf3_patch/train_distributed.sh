#!/bin/bash
NGPUS=$(nvidia-smi -L | wc -l)
if [ "$NGPUS" -lt "2" ]; then
    echo "ERROR: cannot run distributed training without 2 or more GPUs."
    exit 1
fi
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --distributed --cfg "configs/seresnet_unet.yaml"
