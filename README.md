# HomeoLayer

Code for *HomeoLayer: a spiking layer whose homeostatic threshold controller replaces
normalization* (ICLR 2027 submission).

A HomeoLayer is a leaky integrate-and-fire layer with L2-normalized weight rows, a per-feature
affine, a closed-form weight gradient, and a threshold that a controller moves once per timestep
from the layer's own spikes:

    theta <- theta + eta_theta * phi_bar * (r_bar - r*)

`r_bar` is the mean spike count and `phi_bar` the mean one-bit flip statistic over a window of
`T_w` steps, pooled over the layer's neurons, so every sample carries its own threshold and
inference does not depend on the batch. The rate target `r* = sigmoid(rho)` is trained with the
task. No membrane statistics, no running averages, no division at inference.

## Layout

    homeolayer/layers.py      SynapticTrace, HomeoLIF (neuron + controller), HomeoLayer (block)
    homeolayer/model.py       fast-weight attention network and feedforward network
    homeolayer/data.py        SHD, SSC, N-MNIST (tonic) and Google Speech Commands (log-mel)
    homeolayer/experiment.py  hyperparameters of the paper, model of each arm, evaluation
    train.py                  train one arm
    analyze.py                accuracy, firing-rate marginals, learned targets, membrane probe
    transplant/               the controller as a patch for SSM-inspired-LIF (C-SiLIF, RadLIF)
    scripts/                  cache builder, GSC statistics, the full list of runs

## Setup

    pip install -r requirements.txt

Data goes under `data/`: SHD, SSC and N-MNIST are downloaded by tonic on first use; Speech
Commands v0.02 is expected at `data/SpeechCommands/` (torchaudio layout). Frames are cached
under `cache/` on first touch; `python scripts/build_cache.py GSC 32` fills a cache in parallel.

## Training

    python train.py --dataset SHD --arch attn --method homeo --pool layer --seed 0
    python train.py --dataset SSC --arch attn --method bntt --theta 1 --seed 0
    python train.py --dataset NMNIST --arch ff --method none --theta 0.5 --seed 2

| option | values | meaning |
|---|---|---|
| `--dataset` | `SHD` `SSC` `NMNIST` `GSC` | benchmark |
| `--arch` | `attn` `ff` | attention model (two fast-weight attention layers) or feedforward model |
| `--method` | `homeo` `none` `bntt` `ln` | HomeoLayer, fixed threshold, BNTT, LayerNorm |
| `--pool` | `layer` `batch` | controller pooling axis (HomeoLayer) |
| `--fixed-target` | | hold `r*` at its initial value (HomeoLayer) |
| `--one-bit-grad` | | the spike-flip statistic as the weight gradient (ablation) |
| `--theta` | float | the fixed threshold of the standard arms |

Every other setting is fixed at the value of the paper's hyperparameter table
(`homeolayer/experiment.py`). `scripts/reproduce.sh` lists every run behind the tables. The
logged number is the best test accuracy over epochs; checkpoints land in `runs/`.

## Reading a checkpoint

    python analyze.py --dataset SHD --arch attn --method homeo --ckpt runs/SHD_attn_homeo_layer_s0.ck.best
    python analyze.py ... --marginal 10                      # firing-rate marginals of the pooling tables
    python analyze.py ... --probe model.1.snn_attn.ProbLeaky_layer --out probe/   # membrane statistics of one layer

## Transplant

`transplant/homeostasis.patch` adds `--normalization=homeostasis` to
[SSM-inspired-LIF](https://github.com/Maxtimer97/SSM-inspired-LIF) (commit `bca1fbe`) for its
C-SiLIF and RadLIF layers, with the controller's settings of the paper as defaults.
`transplant/run_ports.sh` lists the native and ported runs.

    git clone https://github.com/Maxtimer97/SSM-inspired-LIF && cd SSM-inspired-LIF
    git checkout bca1fbe && git apply /path/to/homeostasis.patch
