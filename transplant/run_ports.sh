#!/bin/bash
# The transplant runs of the paper, in a checkout of SSM-inspired-LIF with homeostasis.patch
# applied. Native rows use the repository's own BatchNorm ("batchnorm"); the port replaces it
# with the HomeoLayer controller ("homeostasis"). Seeds: default, 42, 73.
#   bash transplant/run_ports.sh <path to SSM-inspired-LIF> <gpu index> <path to SHD data> <path to SSC data>
set -e
REPO=$1; GPU=${2:-0}; SHD=$3; SSC=$4
cd "$REPO"
export WANDB_MODE=offline
for SEED in "" 42 73; do
  TAG=${SEED:+_s$SEED}; SEEDARG=${SEED:+--seed $SEED}
  for NORM in batchnorm homeostasis; do
    python -u main.py --config=shd_csilif.yaml --debug --gpu_device=$GPU --log_tofile=False \
      --normalization=$NORM --new_exp_folder=exp/shd_csilif_${NORM}${TAG} --data_folder="$SHD" $SEEDARG
    python -u main.py --config=ssc_csilif_10ms.yaml --debug --gpu_device=$GPU --log_tofile=False \
      --normalization=$NORM --new_exp_folder=exp/ssc_csilif_${NORM}${TAG} --data_folder="$SSC" $SEEDARG
    python -u main.py --config=ssc_csilif_10ms.yaml --debug --model_type=RadLIF --gpu_device=$GPU --log_tofile=False \
      --normalization=$NORM --new_exp_folder=exp/ssc_radlif_${NORM}${TAG} --data_folder="$SSC" $SEEDARG
  done
done
