#!/usr/bin/env python3
"""Fill the on-disk frame cache of a dataset in parallel.

    python scripts/build_cache.py GSC [workers] [--data-dir data] [--cache-dir cache]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_sets = None


def _init(dataset, data_dir, cache_dir):
    global _sets
    import torch
    torch.set_num_threads(1)
    from homeolayer.data import load_data
    tr, te, _, _ = load_data(dataset, 1, 4000, data_dir, cache_dir)
    _sets = (tr.dataset, te.dataset.dataset)


def _touch(job):
    _sets[job[0]][job[1]]
    return 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('workers', type=int, nargs='?', default=16)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--cache-dir', default='cache')
    a = p.parse_args()
    from homeolayer.data import load_data
    tr, te, _, _ = load_data(a.dataset, 1, 4000, a.data_dir, a.cache_dir)
    jobs = [(0, i) for i in range(len(tr.dataset))] + [(1, i) for i in range(len(te.dataset.dataset))]
    print(f'{a.dataset}: {len(jobs)} items, {a.workers} workers', flush=True)
    t0, done = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(a.dataset, a.data_dir, a.cache_dir)) as pool:
        for _ in pool.imap_unordered(_touch, jobs, chunksize=64):
            done += 1
            if done % 5000 == 0:
                print(f'  {done}/{len(jobs)}  {time.time() - t0:.0f}s', flush=True)
    print(f'done: {done} items in {time.time() - t0:.0f}s')
