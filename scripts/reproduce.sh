#!/bin/bash
# Every training run behind the tables of the paper, three seeds each. Pass a GPU as the first
# argument. Runs are sequential; split the list across GPUs as you see fit.
set -e
DEV=${1:-cuda:0}
run () { python train.py --device "$DEV" "$@"; }

for S in 0 1 2; do
  # Table 1 (main), fixed-threshold sensitivity, and the pooling ablation
  for DS in SHD SSC GSC; do
    run --dataset $DS --arch attn --method homeo --pool layer --seed $S
    run --dataset $DS --arch attn --method bntt  --theta 1   --seed $S
    run --dataset $DS --arch attn --method ln    --theta 1   --seed $S
    for TH in 1 0.5 0.25; do run --dataset $DS --arch attn --method none --theta $TH --seed $S; done
  done
  for DS in SHD NMNIST; do
    run --dataset $DS --arch ff --method homeo --pool layer --seed $S
    run --dataset $DS --arch ff --method bntt  --theta 1   --seed $S
    run --dataset $DS --arch ff --method ln    --theta 1   --seed $S
    for TH in 1 0.5 0.25; do run --dataset $DS --arch ff --method none --theta $TH --seed $S; done
  done
  # batch pooling (three seeds on SHD attention; one seed elsewhere for the marginal statistics)
  run --dataset SHD --arch attn --method homeo --pool batch --seed $S
  # the rate target held fixed, and the one-bit statistic as the weight gradient
  run --dataset SHD --arch attn --method homeo --pool layer --fixed-target --seed $S
  run --dataset SHD --arch attn --method homeo --pool layer --one-bit-grad --seed $S
  run --dataset SHD --arch ff   --method homeo --pool layer --one-bit-grad --seed $S
  run --dataset SSC --arch attn --method homeo --pool layer --one-bit-grad --seed $S
  run --dataset GSC --arch attn --method homeo --pool layer --one-bit-grad --seed $S
  run --dataset NMNIST --arch ff --method homeo --pool layer --one-bit-grad --seed $S
done
for DS in SSC GSC; do run --dataset $DS --arch attn --method homeo --pool batch --seed 0; done
for DS in SHD NMNIST; do run --dataset $DS --arch ff --method homeo --pool batch --seed 0; done
