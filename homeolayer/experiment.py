"""Arguments, hyperparameters, model, loaders and evaluation shared by train.py and analyze.py."""
import argparse
import os

import numpy as np
import torch

from .data import CHANNEL_SHIFT, CLIP_MS, INPUT_BIN, gsc_standardizer, load_data, shd_unseen_mask
from .layers import HomeoLIF
from .model import SpikingNet

BIN_MS = 4.0
T_SYN_MS = 5.0
T_MEM_MS = 20.0
T_WINDOW_MS = 5.0
ONSET_MS = 20.0
DP = 0.1
HIDDEN = 700
HEADS, HEAD_DIM = 14, 50
ATTN_LR = 0.01
LR_THRES = 10.0
TARGET_HIDDEN, TARGET_INPUT = 0.10, 0.01
VAR_SURROGATE = 0.1
BOXCAR_WIDTH = 0.5
BNTT_STEPS = 250
BATCH_SIZE = 128
LEARNING_RATE, ADAM_EPS = 1e-3, 1e-16
BLEND_PROB = 0.5
EPOCHS = {'SHD': 150, 'SSC': 60, 'NMNIST': 60, 'GSC': 60}
METHODS = ('homeo', 'none', 'bntt', 'ln')


def steps(ms, minimum=1):
    return max(minimum, int(round(ms / BIN_MS)))


def add_arm_args(p):
    p.add_argument('--dataset', required=True, choices=('SHD', 'SSC', 'NMNIST', 'GSC'))
    p.add_argument('--arch', default='attn', choices=('attn', 'ff'),
                   help='attention model (2 fast-weight attention layers) or feedforward model')
    p.add_argument('--method', default='homeo', choices=METHODS,
                   help='homeo: HomeoLayer; none: fixed threshold; bntt / ln: normalization baselines')
    p.add_argument('--pool', default='layer', choices=('layer', 'batch'),
                   help='homeo: controller pooling axis (layer = one threshold per sample)')
    p.add_argument('--fixed-target', action='store_true', help='homeo: hold r* at its initial value')
    p.add_argument('--one-bit-grad', action='store_true',
                   help='homeo: the spike-flip statistic as the weight gradient (ablation)')
    p.add_argument('--theta', type=float, default=1.0, help='none / bntt / ln: the fixed threshold')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--cache-dir', default='cache')
    return p


def arm_name(a):
    parts = [a.dataset, a.arch]
    if a.method == 'homeo':
        parts += ['homeo', a.pool]
        if a.fixed_target:
            parts.append('fixtgt')
        if a.one_bit_grad:
            parts.append('onebit')
    else:
        parts += [a.method, f'th{a.theta:g}']
    parts.append(f's{a.seed}')
    return '_'.join(parts)


def neuron_settings(a):
    if a.method == 'homeo':
        return dict(thres_init=1.0, thres_window=steps(T_WINDOW_MS), lr_thres=LR_THRES, pool=a.pool,
                    learn_target=not a.fixed_target, affine=True, norm=None,
                    var_surrogate=VAR_SURROGATE, surrogate=None, one_bit_grad=a.one_bit_grad)
    return dict(thres_init=a.theta, thres_window=steps(T_WINDOW_MS), lr_thres=LR_THRES, pool='none',
                learn_target=False, affine=False, norm=None if a.method == 'none' else a.method,
                norm_steps=BNTT_STEPS, var_surrogate=VAR_SURROGATE,
                surrogate='boxcar', surrogate_width=BOXCAR_WIDTH, one_bit_grad=False)


def build_model(a, num_classes, input_width, device):
    torch.manual_seed(a.seed)
    return SpikingNet(
        steps(T_SYN_MS), steps(T_MEM_MS), DP,
        input_size=SpikingNet.binned_size(input_width, INPUT_BIN[a.dataset]),
        hidden_size=HIDDEN, num_heads=HEADS, head_dim=HEAD_DIM, num_classes=num_classes,
        num_attn_layers=2 if a.arch == 'attn' else 0, num_ff_blocks=1,
        attn_lr=ATTN_LR, weight_normalized=(a.method == 'homeo'),
        target_hidden=TARGET_HIDDEN, target_input=TARGET_INPUT,
        input_bin_size=INPUT_BIN[a.dataset], aug_channel_shift=CHANNEL_SHIFT[a.dataset],
        device=device, dtype=torch.float, **neuron_settings(a)).to(device)


class Setup:
    def __init__(self, a):
        self.a = a
        self.device = torch.device(a.device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.train_loader, self.test_loader, self.num_classes, width = load_data(
            a.dataset, BATCH_SIZE, int(BIN_MS * 1000), a.data_dir, a.cache_dir)
        self.model = build_model(a, self.num_classes, width, self.device)
        self.T = steps(CLIP_MS[a.dataset])
        self.t0 = steps(ONSET_MS, 0)
        self.unseen = shd_unseen_mask(a.data_dir, self.test_loader.perm) if a.dataset == 'SHD' else None
        stats = os.path.join(a.data_dir, 'gsc_fbank_stats.npz')
        self.prep = gsc_standardizer(stats, self.device) if a.dataset == 'GSC' else (lambda x: x)

    def forward(self, x):
        return self.model(self.prep(x.to(self.device)), self.T, self.t0)

    @torch.no_grad()
    def evaluate(self, max_batches=None):
        self.model.eval()
        ok = []
        for i, (x, y) in enumerate(self.test_loader):
            if max_batches is not None and i >= max_batches:
                break
            ok.append((self.forward(x).argmax(1).cpu() == y).float())
        ok = torch.cat(ok).numpy()
        acc = 100 * ok.mean()
        if self.unseen is None or max_batches is not None:
            return acc, acc
        return acc, 100 * ok[self.unseen].mean()


def load_checkpoint(model, path, device):
    """Load a checkpoint of train.py; entries the readout never uses are ignored."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck['net']
    keys = set(model.state_dict())
    readouts = tuple(n + '.' for n, m in model.named_modules() if isinstance(m, HomeoLIF) and m.output_layer)
    legacy = {k for k in state if k not in keys and (k.endswith('attn_rms') or k.startswith(readouts))}
    missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if k not in legacy}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return ck
