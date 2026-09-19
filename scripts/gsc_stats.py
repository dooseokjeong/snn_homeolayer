#!/usr/bin/env python3
"""Per-bin mean and standard deviation of the GSC log-mel features over 2,000 training clips
(data/gsc_fbank_stats.npz).

    python scripts/gsc_stats.py [--clips 2000] [--data-dir data] [--out data/gsc_fbank_stats.npz]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from homeolayer.data import GSCFbank

p = argparse.ArgumentParser()
p.add_argument('--clips', type=int, default=2000)
p.add_argument('--data-dir', default='data')
p.add_argument('--out', default='data/gsc_fbank_stats.npz')
a = p.parse_args()
ds = GSCFbank(a.data_dir, 'training', 4.0)
idx = np.random.default_rng(0).choice(len(ds), a.clips, replace=False)
frames = np.concatenate([ds[int(i)][0] for i in idx], axis=0)
np.savez(a.out, mean=frames.mean(0), std=frames.std(0), n_clips=a.clips, frame_shift_ms=4.0)
print(f'wrote {a.out}: {frames.shape[0]} frames from {a.clips} clips, mean {frames.mean():.3f}, std {frames.std():.3f}')
