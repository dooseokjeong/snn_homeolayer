#!/usr/bin/env python3
"""Train one arm of the paper and log its test accuracy after every epoch.

    python train.py --dataset SHD --arch attn --method homeo --pool layer --seed 0
    python train.py --dataset SHD --arch attn --method bntt --theta 1 --seed 0
    python train.py --dataset GSC --arch attn --method none --theta 0.5 --seed 1

Checkpoints go to <out>/<arm>.ck (last epoch, resumed automatically) and <out>/<arm>.ck.best
(best accuracy on the unseen speakers of SHD, best test accuracy elsewhere).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homeolayer.experiment import (ADAM_EPS, BLEND_PROB, EPOCHS, LEARNING_RATE, Setup, add_arm_args,
                                   arm_name)


def main():
    p = add_arm_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--epochs', type=int, default=None, help='default: 150 on SHD, 60 elsewhere')
    p.add_argument('--out', default='runs')
    a = p.parse_args()
    epochs = a.epochs or EPOCHS[a.dataset]
    name = arm_name(a)
    os.makedirs(a.out, exist_ok=True)
    ck_path = os.path.join(a.out, name + '.ck')

    s = Setup(a)
    model = s.model
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
    crit = nn.CrossEntropyLoss()
    hist = {'acc': [], 'unseen': [], 'train': []}
    start = 0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=s.device, weights_only=False)
        model.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        hist, start = ck['hist'], ck['epoch'] + 1
        print(f'[resume] {ck_path} at epoch {start}', flush=True)
    print(f'=== {name}: {sum(q.numel() for q in model.parameters()):,} parameters, T={s.T}, '
          f'{len(s.train_loader)} training batches, {epochs} epochs', flush=True)

    best_acc = max(hist['acc'], default=0.0)
    best_unseen = max(hist['unseen'], default=0.0)
    t_start = time.time()
    for ep in range(start, epochs):
        model.train()
        losses = []
        for x, y in s.train_loader:
            opt.zero_grad()
            x = s.prep(x.to(s.device))
            x, _ = model.blend_batch(x, y.to(s.device), BLEND_PROB)
            loss = crit(model(x, s.T, s.t0).cpu(), y.long())
            loss.backward()
            opt.step()
            losses.append(loss.item())
        acc, unseen = s.evaluate()
        improved = unseen > best_unseen
        best_acc, best_unseen = max(best_acc, acc), max(best_unseen, unseen)
        hist['acc'].append(acc); hist['unseen'].append(unseen); hist['train'].append(float(np.mean(losses)))
        state = dict(net=model.state_dict(), opt=opt.state_dict(), epoch=ep, hist=hist, acc=acc, unseen=unseen, arm=name)
        torch.save(state, ck_path)
        if improved:
            torch.save(state, ck_path + '.best')
        print(f'  ep {ep:3d}  train_loss={np.mean(losses):.4f}  test={acc:6.2f}%  unseen={unseen:6.2f}%  '
              f'(best {best_acc:.2f})  ({time.time() - t_start:.0f}s)', flush=True)
    print(f'RESULT {name}  best_test={best_acc:.2f}%  best_unseen={best_unseen:.2f}%', flush=True)


if __name__ == '__main__':
    main()
